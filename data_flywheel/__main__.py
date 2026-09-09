"""
BettaFish 数据飞轮 CLI 入口

允许开发者直接在命令行运行爬虫与清洗流水线:
python -m data_flywheel
"""

import sys
import logging
import argparse

from data_flywheel.betta_spider import BettaSpider

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="BettaFish Data Flywheel")
    parser.add_argument("--mock", action="store_true", help="Force only mock source to easily test the pipeline offline")
    parser.add_argument("--loop", action="store_true", help="Run endlessly in a loop")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (if looping)")
    
    args = parser.parse_args()
    
    config = {
        "enable_mock_source": True,
        "rss_feeds": [] if args.mock else ["https://rsshub.app/cls/telegraph"]
    }
    
    spider = BettaSpider(config_paths=config)
    
    if args.loop:
        spider.run_loop(interval_seconds=args.interval)
    else:
        spider.run_once(max_articles=5)
        print("\n--- 最新生成的 SeedEvents ---")
        events = spider.store.read_latest(2)
        for e in events:
             print(f"[{e.sentiment_label}] {e.title} -> Impact: {e.impact_level}")
             print("  [to_policy_input() 预览]")
             print("  " + e.to_policy_input().replace("\n", "\n  "))
             print("-" * 40)
             
if __name__ == "__main__":
    main()
