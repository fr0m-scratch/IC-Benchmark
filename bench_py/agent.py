#!/usr/bin/env python3
import argparse
import os
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # lightweight fallback if python-dotenv isn't available
    def load_dotenv(path: str = '.env'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    if '=' in s:
                        k, v = s.split('=', 1)
                        k = k.strip(); v = v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
        except FileNotFoundError:
            pass
from bench_py.agent_executor import AgentConfig, AgentExecutor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--provider', type=str, default='ollama', choices=['ollama','gemini'])
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--prompt', type=str, required=True)
    p.add_argument('--server', type=str, default='http://localhost:3000')
    p.add_argument('--ollama', type=str, default='http://localhost:11434')
    p.add_argument('--gemini-api-key', type=str, default=None)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=4)
    p.add_argument('--retry', type=int, default=1)
    return p.parse_args()


def main():
    # load .env if available
    if load_dotenv is not None:
        load_dotenv()
    ns = parse_args()
    api_key = ns.gemini_api_key or os.environ.get('GEMINI_API_KEY')
    cfg = AgentConfig(
        server=ns.server,
        provider=ns.provider,
        model=ns.model,
        ollama=ns.ollama,
        gemini_api_key=api_key,
        limit_tools=ns.limit,
        max_steps=ns.max_steps,
        retry_per_step=ns.retry,
    )
    ex = AgentExecutor(cfg)
    result = ex.run(ns.prompt)
    print('\n--- RESULT ---')
    print(result)


if __name__ == '__main__':
    main()
