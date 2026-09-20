#!/usr/bin/env python3
"""Small deterministic quality rejection screen, NEVER a 97% retention certificate.

Requires aiohttp. Normal stopping is used. All complete HTTP requests, raw
responses, expected answers, and scoring decisions are retained beside --out.
Optional --long-context uses the server's actual nonthinking chat template,
exactly 131072 input tokens, and three unique key/value records near 5/50/95%.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

NOTICE = ('This is a small deterministic rejection screen, not an accuracy benchmark, '
          'base-model comparison, or certification of 97% accuracy retention. A pass '
          'only establishes correctness on these examples under this configuration.')
SYSTEM = 'Follow the requested output format exactly. Answer directly without explanations or reasoning.'
LONG_TOKENS = 131072


def short_cases():
    cases = []
    for name, question, answer in (
        ('arithmetic_multiply', 'What is 17 multiplied by 23? Return only the integer.', '391'),
        ('arithmetic_add', 'Compute 1587 + 2468. Return only the integer.', '4055'),
        ('arithmetic_subtract', 'Compute 997 - 428. Return only the integer.', '569'),
        ('arithmetic_precedence', 'Compute 144 / 12 + 7 * 3 using ordinary arithmetic precedence. Return only the integer.', '33'),
        ('logic_entailment', 'All sparrows are birds. Some birds can fly. Do these two statements logically imply that every sparrow can fly? Return only YES or NO.', 'NO'),
        ('logic_order', 'Dana arrived before Eli. Eli arrived before Farah. Among these three people, who arrived first? Return only the name.', 'Dana'),
        ('extract_identifier', 'Use only this passage: The ALDER shipment has tracking code RX-4821. The BIRCH shipment has tracking code LM-7306. What is the BIRCH tracking code? Return only the code.', 'LM-7306'),
        ('extract_place', 'Use only this passage: The meeting was first planned for Cedar Hall. An update moved it to Maple Room. A later notice explicitly confirmed Maple Room without changes. What is the final meeting location? Return only the location.', 'Maple Room'),
        ('code_sum', 'What exactly does this Python 3 program print? Return only its output.\nprint(sum(x * x for x in range(5)))', '30'),
        ('code_loop', 'What exactly does this Python 3 program print? Return only its output.\nparts = []\nfor i in range(1, 5):\n    if i % 2 == 0:\n        parts.append(str(i * 3))\nprint("-".join(parts))', '6-12'),
    ):
        cases.append({'name': name, 'prompt': question, 'expected': answer, 'scorer': 'exact_text'})
    cases.extend([
        {'name': 'json_extract', 'prompt': 'Return one JSON object with exactly these fields: name is the string Mira, count is the integer 7, active is the boolean false. No other text.',
         'expected': {'name': 'Mira', 'count': 7, 'active': False}, 'scorer': 'strict_json',
         'schema': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'count': {'type': 'integer'}, 'active': {'type': 'boolean'}},
                    'required': ['name', 'count', 'active'], 'additionalProperties': False}},
        {'name': 'json_transform', 'prompt': 'Sort the integers [9, 2, 9, 1] in ascending order, keeping duplicates. Return only a JSON object with exactly two fields: sorted is that integer array; distinct is the number of distinct input integers.',
         'expected': {'sorted': [1, 2, 9, 9], 'distinct': 3}, 'scorer': 'strict_json',
         'schema': {'type': 'object', 'properties': {'sorted': {'type': 'array', 'items': {'type': 'integer'}}, 'distinct': {'type': 'integer'}},
                    'required': ['sorted', 'distinct'], 'additionalProperties': False}},
    ])
    return cases


def schema_errors(value, schema, path='$'):
    """Validate the deliberately small schema subset used by this screen."""
    expected_type = schema.get('type')
    actual_types = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool}
    if expected_type and type(value) is not actual_types[expected_type]:
        return [f'{path}: expected {expected_type}, got {type(value).__name__}']
    errors = []
    if expected_type == 'object':
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(f'{path}: missing required key {key}')
        if schema.get('additionalProperties') is False:
            errors.extend(f'{path}: unexpected key {key}' for key in value if key not in properties)
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(item, properties[key], f'{path}.{key}'))
    elif expected_type == 'array':
        for index, item in enumerate(value):
            errors.extend(schema_errors(item, schema.get('items', {}), f'{path}[{index}]'))
    return errors


def score_answer(case, content):
    if not isinstance(content, str):
        return {'pass': False, 'reason': 'Missing string answer', 'observed': content}
    stripped = content.strip()
    if case['scorer'] == 'exact_text':
        passed = stripped == case['expected']
        return {'pass': passed, 'rule': 'Exact text after trimming outer whitespace; explanations and code fences fail.',
                'observed': stripped, 'expected': case['expected'],
                'reason': 'match' if passed else 'answer or format mismatch'}
    try:
        observed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return {'pass': False, 'reason': f'Not a standalone valid JSON value: {exc}', 'observed': stripped}
    errors = schema_errors(observed, case['schema'])
    passed = not errors and observed == case['expected']
    return {'pass': passed, 'rule': 'Standalone JSON, exact schema types/keys, and exact expected values.',
            'observed': observed, 'expected': case['expected'], 'schema_errors': errors,
            'reason': 'match' if passed else 'schema or value mismatch'}


def endpoint(base, path):
    base = base.rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    return base + path


async def json_api(session, args, path, payload):
    async with session.post(endpoint(args.base_url, path), json=payload) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f'{path} HTTP {response.status}: {text[:1500]}')
        return json.loads(text)


async def tokenize(session, args, text):
    return await json_api(session, args, '/tokenize',
                          {'model': args.model, 'prompt': text, 'add_special_tokens': False})


async def chat_envelope(session, args):
    marker = 'QUALITYBODYMARKERf9238c71a64b5e02END'
    body_ids = (await tokenize(session, args, marker))['tokens']
    request = {'model': args.model, 'messages': [{'role': 'system', 'content': SYSTEM},
               {'role': 'user', 'content': marker}], 'add_generation_prompt': True,
               'chat_template_kwargs': {'enable_thinking': False}}
    rendered = await json_api(session, args, '/tokenize', request)
    ids = rendered['tokens']
    occurrences = [index for index in range(len(ids) - len(body_ids) + 1)
                   if ids[index:index + len(body_ids)] == body_ids]
    if len(occurrences) != 1:
        raise RuntimeError('Cannot isolate the user body in the actual server chat template; refusing to invent a template')
    start = occurrences[0]
    return ids[:start], ids[start + len(body_ids):], rendered.get('max_model_len'), request


def make_distractors(seed, target_chars):
    rng = random.Random(seed)
    pieces, length, index = [], 0, 0
    sites = ['cedar grove', 'eastern harbor', 'north field', 'western ridge', 'south garden']
    while length < target_chars:
        index += 1
        text = (f'Routine inventory record D{index}: item tag {rng.getrandbits(48):012x}; '
                f'location {rng.choice(sites)}; count {rng.randrange(100, 9999)}; '
                f'audit reference {rng.getrandbits(40):010x}. '
                'This ordinary inventory observation is not an authoritative lookup entry.\n')
        pieces.append(text)
        length += len(text)
    return ''.join(pieces)


async def long_case(session, args, artifacts):
    prefix, suffix, max_len, template_request = await chat_envelope(session, args)
    if max_len and max_len < LONG_TOKENS + args.long_output_tokens:
        raise RuntimeError(f'Long screen requires {LONG_TOKENS + args.long_output_tokens} tokens, server max is {max_len}')
    rng = random.Random(args.seed)
    expected = {f'KEY_{rng.getrandbits(48):012x}': f'VALUE_{rng.getrandbits(48):012x}' for _ in range(3)}
    header = ('The following archive contains routine inventory records and three authoritative lookup entries. '
              'Read the entire archive. At the end, return the requested lookup values exactly, using only the authoritative entries.\n')
    question = ('\nEND OF ARCHIVE. Return only one JSON object mapping each of these keys to its exact lookup value: '
                + ', '.join(expected) + '. Include all three keys, and no extra keys or text.\n')
    header_ids = (await tokenize(session, args, header))['tokens']
    question_ids = (await tokenize(session, args, question))['tokens']
    target_chars = LONG_TOKENS * 6
    while True:
        filler = make_distractors(args.seed + 9001, target_chars)
        filler_ids = (await tokenize(session, args, filler))['tokens']
        if len(filler_ids) >= LONG_TOKENS:
            break
        target_chars *= 2
    ids, cursor, positions = prefix + header_ids, 0, []
    for fraction, (key, value) in zip((0.05, 0.50, 0.95), expected.items()):
        target = int(LONG_TOKENS * fraction)
        needed = target - len(ids)
        if needed < 0:
            raise RuntimeError('Template/header unexpectedly exceeds needle position')
        ids.extend(filler_ids[cursor:cursor + needed])
        cursor += needed
        needle = f'\nAUTHORITATIVE LOOKUP ENTRY: key {key} means value {value}.\n'
        needle_ids = (await tokenize(session, args, needle))['tokens']
        positions.append({'key': key, 'value': value, 'needle_start_token': len(ids),
                          'needle_tokens': len(needle_ids), 'fraction_of_input': len(ids) / LONG_TOKENS})
        ids.extend(needle_ids)
    needed = LONG_TOKENS - len(ids) - len(question_ids) - len(suffix)
    if needed < 0:
        raise RuntimeError('Insufficient context budget for final question and assistant generation prefix')
    ids.extend(filler_ids[cursor:cursor + needed])
    ids.extend(question_ids)
    ids.extend(suffix)
    assert len(ids) == LONG_TOKENS
    layout = {'input_tokens': len(ids), 'server_max_model_len': max_len,
              'chat_prefix_tokens': len(prefix), 'assistant_suffix_tokens': len(suffix),
              'template_probe_request': template_request, 'needles': positions,
              'question': question, 'synthetic_long_context': True,
              'prompt_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    layout_path = artifacts / 'long_context_layout.json'
    layout_path.write_text(json.dumps(layout, indent=2) + '\n')
    return {'name': 'long_context_131072', 'token_ids': ids, 'expected': expected,
            'scorer': 'strict_json', 'schema': {'type': 'object',
                'properties': {key: {'type': 'string'} for key in expected},
                'required': list(expected), 'additionalProperties': False},
            'layout_file': str(layout_path.resolve())}


async def run_case(session, args, artifacts, case):
    long = 'token_ids' in case
    payload = {'model': args.model, 'max_tokens': args.long_output_tokens if long else args.max_tokens,
               'temperature': 0, 'seed': args.seed, 'stream': False}
    if long:
        payload.update(prompt=case['token_ids'], add_special_tokens=False)
        path = '/v1/completions'
    else:
        payload.update(messages=[{'role': 'system', 'content': SYSTEM},
                                 {'role': 'user', 'content': case['prompt']}],
                       chat_template_kwargs={'enable_thinking': False})
        path = '/v1/chat/completions'
        if 'schema' in case:
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': case['name'], 'strict': True, 'schema': case['schema']}}
    request_path = artifacts / (case['name'] + '.request.json')
    request_path.write_text(json.dumps({'endpoint': endpoint(args.base_url, path), 'body': payload}, indent=2) + '\n')
    result = {'name': case['name'], 'expected': case['expected'], 'scorer': case['scorer'],
              'schema': case.get('schema'), 'request_file': str(request_path.resolve()),
              'layout_file': case.get('layout_file'), 'http_status': None, 'response': None,
              'scoring': {'pass': False, 'reason': 'Request not completed'}}
    started = time.monotonic()
    try:
        timeout = aiohttp.ClientTimeout(total=args.long_timeout if long else args.timeout)
        async with session.post(endpoint(args.base_url, path), json=payload, timeout=timeout) as response:
            raw = await response.text()
            raw_path = artifacts / (case['name'] + '.response.txt')
            raw_path.write_text(raw)
            result.update(http_status=response.status, raw_response_file=str(raw_path.resolve()))
            try:
                body = json.loads(raw)
                result['response'] = body
            except json.JSONDecodeError as exc:
                raise RuntimeError('Response is not valid JSON') from exc
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}')
            choices = body.get('choices') or []
            if len(choices) != 1:
                raise RuntimeError('Expected exactly one response choice')
            choice = choices[0]
            content = choice.get('text') if long else (choice.get('message') or {}).get('content')
            result['scoring'] = score_answer(case, content)
            result['finish_reason'] = choice.get('finish_reason')
            if choice.get('finish_reason') == 'length':
                result['scoring'].update({'pass': False, 'reason': 'Output budget exhausted'})
            if long:
                observed = (body.get('usage') or {}).get('prompt_tokens')
                result['prompt_tokens_verified'] = observed == LONG_TOKENS
                if not result['prompt_tokens_verified']:
                    result['scoring'].update({'pass': False, 'reason': f'Expected exactly {LONG_TOKENS} prompt tokens; usage reported {observed}'})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['scoring'] = {'pass': False, 'reason': result['error']}
    result['seconds'] = time.monotonic() - started
    print(f"{result['name']}: {'PASS' if result['scoring']['pass'] else 'FAIL'} — {result['scoring']['reason']}", flush=True)
    return result


async def main(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = out.with_name(out.stem + '_artifacts')
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {'notice': NOTICE, 'not_accuracy_certification': True, 'base_reference_evaluated': False,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'configuration': vars(args),
              'screen_pass': False, 'results': [], 'preparation_errors': []}
    headers = {}
    if os.environ.get(args.api_key_env):
        headers['Authorization'] = 'Bearer ' + os.environ[args.api_key_env]
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        for case in short_cases():
            report['results'].append(await run_case(session, args, artifacts, case))
            out.write_text(json.dumps(report, indent=2) + '\n')
        if args.long_context:
            try:
                case = await long_case(session, args, artifacts)
                report['results'].append(await run_case(session, args, artifacts, case))
            except Exception as exc:
                report['preparation_errors'].append(f'long context: {type(exc).__name__}: {exc}')
    expected_count = 13 if args.long_context else 12
    passed = sum(result['scoring']['pass'] for result in report['results'])
    report.update(screen_pass=not report['preparation_errors'] and passed == expected_count,
                  passed=passed, required=expected_count,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'out': str(out.resolve()), 'passed': passed, 'required': expected_count,
                      'screen_pass': report['screen_pass'], 'preparation_errors': report['preparation_errors'],
                      'notice': NOTICE}, indent=2))
    return 0 if report['screen_pass'] else 2


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='qwen-lab')
    parser.add_argument('--out', required=True)
    parser.add_argument('--long-context', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--long-output-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--long-timeout', type=float, default=1800)
    parser.add_argument('--seed', type=int, default=7319)
    parser.add_argument('--api-key-env', default='VLLM_API_KEY')
    args = parser.parse_args()
    if min(args.max_tokens, args.long_output_tokens, args.timeout, args.long_timeout) <= 0:
        parser.error('Token budgets and timeouts must be positive')
    return args


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main(arguments())))
