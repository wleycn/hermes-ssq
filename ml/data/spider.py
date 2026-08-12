"""
双色球预测系统 - 数据爬虫模块
从东方财富网爬取双色球历史开奖数据，支持多页爬取、增量保存为CSV文件。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests as req
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from lxml import etree
import time
import random
from collections import deque
import pandas as pd
from datetime import datetime

from ml.config import SPIDER_CONFIG


class SsqSpider:
    """双色球历史数据爬虫

    从东方财富网爬取双色球历史开奖数据，支持多页爬取，
    自动解析HTML并增量保存为CSV文件。

    Attributes:
        spider_name: 爬虫名称，用于文件命名
        max_pages: 最大爬取页数
        save_dir: 保存目录路径
        url_template: URL模板
        session: HTTP会话对象
        url_queue: 待爬取URL队列
        all_data: 已爬取的所有数据
    """

    def __init__(self, spider_name: str = "ssq_history", max_pages: int = 1, save_dir: str | None = None):
        """初始化爬虫配置

        Args:
            spider_name: 爬虫名称，用于文件命名，默认 'ssq_history'
            max_pages: 最大爬取页数，默认 1
            save_dir: 保存目录，默认为配置文件中的 default_save_dir
        """
        self.spider_name = spider_name
        self.max_pages = max_pages
        self.save_dir = Path(save_dir) if save_dir else Path(SPIDER_CONFIG["default_save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.url_template = SPIDER_CONFIG["target_url"]
        self.session = self._create_session()
        self.url_queue: deque[str] = deque()
        self.all_data: list[list[str]] = []

    def _create_session(self) -> req.Session:
        """创建配置好重试策略的HTTP会话

        Returns:
            配置好的 requests.Session 对象
        """
        session = req.Session()
        session.headers.update({"user-agent": random.choice(SPIDER_CONFIG["user_agents"])})

        retries = Retry(
            total=SPIDER_CONFIG["retry_total"],
            backoff_factor=SPIDER_CONFIG["retry_backoff_factor"],
            status_forcelist=SPIDER_CONFIG["retry_status_codes"]
        )
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        return session

    def build_url_queue(self) -> None:
        """构建待爬取的URL队列"""
        for page in range(1, self.max_pages + 1):
            self.url_queue.append(self.url_template.format(page))
        print(f"[URL构建] 成功生成 {len(self.url_queue)} 个待爬取链接")

    def fetch_page(self, url: str) -> str | None:
        """发起HTTP请求获取页面内容

        Args:
            url: 目标URL

        Returns:
            HTML文本，失败返回None
        """
        print(f"[请求中] {url}")
        try:
            time.sleep(random.uniform(SPIDER_CONFIG["request_delay_min"], SPIDER_CONFIG["request_delay_max"]))
            response = self.session.get(url, timeout=SPIDER_CONFIG["request_timeout"])
            response.raise_for_status()
            return response.text
        except req.exceptions.RequestException as e:
            print(f"[请求异常] {url} -> {e}")
            return None

    def parse_data(self, html_str: str) -> list[list[str]]:
        """解析HTML页面提取开奖数据

        Args:
            html_str: HTML字符串

        Returns:
            解析后的行数据列表
        """
        if not html_str:
            return []

        try:
            tree = etree.HTML(html_str)
            rows = tree.xpath('//tbody/tr')
            page_data: list[list[str]] = []

            for row in rows:
                row_data: list[str] = []
                cells = row.xpath('./td')

                for i, cell in enumerate(cells[:4]):
                    if i == 2:
                        continue

                    text = cell.xpath('string(.)').strip()

                    if i == 1:
                        date_str = text.split('(')[0].strip()
                        date_parts = date_str.split('-')
                        date_parts.pop()
                        row_data.extend(date_parts + [date_str])
                        continue

                    if i == 3:
                        spans = [s.strip() for s in cell.xpath('.//span/text()')]
                        row_data.extend(spans)
                        continue

                    row_data.append(text)

                if len(row_data) == len(SPIDER_CONFIG["csv_headers"]):
                    page_data.append(row_data)

            print(f"[解析成功] 提取到 {len(page_data)} 条记录")
            return page_data

        except Exception as e:
            print(f"[解析异常] {e}")
            return []

    def save_html(self, html_str: str, page_num: int) -> None:
        """保存HTML到本地文件（调试用）

        Args:
            html_str: HTML字符串
            page_num: 页码
        """
        if not html_str:
            print(f"[保存跳过] 第 {page_num} 页未获取到有效内容")
            return

        file_path = self.save_dir / f"{self.spider_name}-page{page_num}.html"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            print(f"[保存成功] 页面已保存至: {file_path}")
        except IOError as e:
            print(f"[保存失败] 写入文件时发生错误: {e}")

    def _get_existing_dnums(self, csv_path: Path) -> set[str]:
        """获取已存在CSV文件中最后几行的dNum列表

        Args:
            csv_path: CSV文件路径

        Returns:
            已存在的dNum集合（字符串格式）
        """
        if not csv_path.exists():
            return set()

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            if len(df) == 0:
                return set()

            last_df = df.tail(SPIDER_CONFIG["incremental_check_rows"])
            existing_dnums = set(last_df.iloc[:, 0].astype(str).tolist())
            print(f"[增量检查] 已读取最后 {len(last_df)} 条记录的dNum")
            return existing_dnums
        except Exception as e:
            print(f"[读取失败] 无法读取已有CSV文件: {e}")
            return set()

    def _clean_trailing_empty_lines(self, csv_path: Path) -> None:
        """清理CSV文件末尾的空行

        Args:
            csv_path: CSV文件路径
        """
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()

            content = content.rstrip('\n') + '\n'

            with open(csv_path, 'w', encoding='utf-8-sig') as f:
                f.write(content)

            print("[清理空行] 已确保文件末尾没有多余空行")
        except Exception as e:
            print(f"[清理空行失败] {e}")

    def save_csv(self, filename: str | None = None) -> str | None:
        """增量保存爬取数据到CSV文件

        检查已有数据的最后INCREMENTAL_CHECK_ROWS行，只追加新增的记录。
        新增数据按dNum升序排序后从尾部插入。

        Args:
            filename: 自定义文件名，默认为配置中的 csv_filename

        Returns:
            保存的文件路径，无新增数据时返回None
        """
        if not self.all_data:
            print("[保存跳过] 未获取到任何数据")
            return None

        csv_path = self.save_dir / (filename or SPIDER_CONFIG["csv_filename"])
        existing_dnums = self._get_existing_dnums(csv_path)

        new_data: list[list[str]] = []
        for row in self.all_data:
            dnum = str(row[0]).strip()
            if dnum not in existing_dnums:
                new_data.append(row)

        if not new_data:
            print("[保存跳过] 没有新增数据")
            return None

        try:
            csv_headers = SPIDER_CONFIG["csv_headers"]
            df = pd.DataFrame(new_data, columns=csv_headers)
            df[csv_headers[0]] = pd.to_numeric(df[csv_headers[0]], errors='coerce').fillna(0).astype(int)
            df = df.sort_values(by=csv_headers[0], ascending=True)

            file_exists = csv_path.exists()

            if file_exists:
                self._clean_trailing_empty_lines(csv_path)

            df.to_csv(
                csv_path,
                index=False,
                encoding='utf-8-sig',
                mode='a',
                header=not file_exists
            )

            print(f"\n[CSV保存] 成功追加 {len(new_data)} 条新记录到: {csv_path}")
            print(f"[排序信息] 新增数据已按dNum升序排列")
            return str(csv_path)
        except Exception as e:
            print(f"[CSV保存失败] {e}")
            return None

    def run(self, save_html: bool = False) -> list[list[str]]:
        """爬虫主调度方法

        Args:
            save_html: 是否保存HTML文件（调试用），默认 False

        Returns:
            所有爬取的数据
        """
        print("=" * 60)
        print(f"SpiderGo: {datetime.now()}")
        print("=" * 60)

        self.build_url_queue()
        page_count = 0

        while self.url_queue:
            url = self.url_queue.popleft()
            page_count += 1

            html_str = self.fetch_page(url)

            if save_html:
                self.save_html(html_str, page_count)

            page_data = self.parse_data(html_str)
            self.all_data.extend(page_data)

        print(f"\n[爬取完成] 共爬取 {page_count} 页，获取 {len(self.all_data)} 条记录")
        self.save_csv()

        print("=" * 60)
        print("任务完成！")
        print("=" * 60)

        return self.all_data


if __name__ == "__main__":
    spider = SsqSpider(
        spider_name="ssq_history",
        max_pages=1
    )
    spider.run()