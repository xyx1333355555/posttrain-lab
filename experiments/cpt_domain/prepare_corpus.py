"""E1 · 垂直领域 CPT 语料构建（v3）

垂直领域 = 密码学与 AI 安全
  领域-英文：arXiv cs.CR（Cryptography and Security）摘要
  领域-中文：中文维基「密码学 / 网络安全 / AI 安全 / 大模型」类条目
  通用回放：中英文维基正文（确定性分页抽样，非随机，避免命中率过低）

特性：每一步结果本地缓存（D:/data/cpt_domain/_cache），重跑不重复抓取。

用法（需绕过失效代理 + 放开沙箱）：
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy python experiments/cpt_domain/prepare_corpus.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0 (research-corpus-builder; contact: student)"}

ARXIV_API = "https://export.arxiv.org/api/query"
WIKI = {"zh": "https://zh.wikipedia.org/w/api.php", "en": "https://en.wikipedia.org/w/api.php"}

ZH_TITLES = [
    "密码学", "密码学历史", "对称密钥加密", "公开密钥加密", "分组密码", "流密码",
    "高级加密标准", "数据加密标准", "RSA加密算法", "椭圆曲线密码学", "迪菲-赫尔曼密钥交换",
    "哈希函数", "SHA-2", "消息认证码", "HMAC", "数字签名", "数字证书", "公钥基础设施",
    "认证机构", "密钥管理", "一次性密码本", "密码学安全伪随机数生成器", "分组密码工作模式",
    "运输层安全", "TLS", "SSH", "IPsec", "Kerberos", "虚拟专用网", "洋葱路由",
    "超文本传输安全协议", "HTTP严格传输安全",
    "密码分析", "暴力破解攻击", "字典攻击", "中间人攻击", "重放攻击", "侧信道攻击",
    "时序攻击", "功耗分析", "选择明文攻击", "选择密文攻击", "已知明文攻击", "唯密文攻击",
    "生日攻击", "碰撞攻击", "彩虹表", "社会工程学", "钓鱼式攻击", "拒绝服务攻击",
    "分布式拒绝服务攻击", "缓冲区溢出", "SQL注入", "跨站脚本", "跨站请求伪造",
    "文件包含漏洞", "提权", "后门", "计算机病毒", "蠕虫病毒", "勒索软件", "僵尸网络",
    "高级持续性威胁", "零日攻击", "计算机安全漏洞", "通用漏洞披露", "渗透测试", "红队", "蜜罐",
    "计算机安全", "网络安全", "信息安全", "防火墙", "入侵检测系统", "入侵防御系统",
    "访问控制", "最小权限原则", "纵深防御", "安全审计", "可信计算", "可信执行环境",
    "沙盒 (计算机安全)", "安全多方计算", "零知识证明", "同态加密", "差分隐私", "隐私计算",
    "联邦学习", "硬件安全模块",
    "量子密钥分发", "量子密码学", "后量子密码学", "格密码", "基于身份的加密", "属性基加密",
    "区块链", "智能合约", "加密货币", "比特币",
    "人工智能安全", "人工智能对齐", "人工智能伦理", "对抗样本", "对抗攻击",
    "模型窃取", "数据投毒", "提示注入", "大语言模型", "生成式预训练变换器",
    "Transformer模型", "注意力机制", "基于人类反馈的强化学习", "思维链", "检索增强生成",
    "可解释人工智能", "模型鲁棒性",
]


def _get(url: str, retries: int = 4, timeout: int = 40) -> bytes:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout, context=CTX) as r:
                return r.read()
        except urllib.error.HTTPError as e:  # noqa: F821
            last = e
            if e.code == 429:  # 限流：长退避
                time.sleep(8 + 6 * i)
                continue
            time.sleep(1.5 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed: {last}")


def clean(t: str) -> str:
    t = re.sub(r"\$[^$]{0,80}\$", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def cache_load(path: str):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def cache_save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def arxiv_docs(total: int, cat: str = "cs.CR", per: int = 100) -> list[str]:
    docs, seen, start = [], set(), 0
    ns = {"a": "http://www.w3.org/2005/Atom"}
    while len(docs) < total:
        q = urllib.parse.urlencode({
            "search_query": f"cat:{cat}", "start": start,
            "max_results": min(per, total - len(docs)),
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        try:
            root = ET.fromstring(_get(f"{ARXIV_API}?{q}"))
        except Exception as e:  # noqa: BLE001
            print(f"  [arxiv] 中断 start={start}: {str(e)[:70]}", flush=True)
            break
        ents = root.findall("a:entry", ns)
        if not ents:
            break
        for e in ents:
            title = clean(e.findtext("a:title", "", ns) or "")
            summ = clean(e.findtext("a:summary", "", ns) or "")
            if len(summ) < 200 or title.lower()[:70] in seen:
                continue
            seen.add(title.lower()[:70])
            docs.append(f"{title}. {summ}")
        start += len(ents)
        print(f"  [arxiv] {len(docs)} 篇", flush=True)
        time.sleep(1.0)
    return docs[:total]


def wiki_titles(lang: str, titles: list[str]) -> list[str]:
    """按标题批量取正文（每次 10 个标题，串行 + 退避）。"""
    docs, seen = [], set()
    for i in range(0, len(titles), 10):
        batch = titles[i:i + 10]
        params = {"action": "query", "format": "json", "prop": "extracts",
                  "explaintext": 1, "redirects": 1, "titles": "|".join(batch)}
        try:
            js = json.loads(_get(f"{WIKI[lang]}?{urllib.parse.urlencode(params)}").decode("utf-8", "ignore"))
        except Exception as e:  # noqa: BLE001
            print(f"  [wiki-{lang}] 批次 {i//10+1} 失败: {str(e)[:50]}", flush=True)
            continue
        for p in js.get("query", {}).get("pages", {}).values():
            ext, title = clean(p.get("extract") or ""), p.get("title") or ""
            if len(ext) > 500 and title not in seen:
                seen.add(title)
                docs.append(ext)
        time.sleep(1.2)
    return docs


def wikitext_docs(n: int, name: str = "wikitext-2-raw-v1") -> list[str]:
    """通用英文语料：WikiText（英文维基正文，走 HF 国内镜像）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from datasets import load_dataset  # 延迟导入，避免无网络时影响其他步骤

    ds = load_dataset("Salesforce/wikitext", name, split="train")
    docs: list[str] = []
    for t in ds["text"]:
        t = re.sub(r"\s+", " ", t.strip())
        if len(t) > 200:
            docs.append(t)
        if len(docs) >= n:
            break
    return docs


def wiki_allpages(lang: str, n: int) -> list[str]:
    """确定性分页取正文（从指定前缀向后走），每请求最多 20 篇。"""
    if n <= 0:
        return []
    docs, seen, gapfrom = [], set(), ""
    for _ in range(400):
        if len(docs) >= n:
            break
        params = {
            "action": "query", "format": "json", "generator": "allpages",
            "gapnamespace": 0, "gaplimit": 50, "gapfrom": gapfrom,
            "prop": "extracts", "explaintext": 1, "exlimit": "max",
        }
        try:
            js = json.loads(_get(f"{WIKI[lang]}?{urllib.parse.urlencode(params)}").decode("utf-8", "ignore"))
        except Exception as e:  # noqa: BLE001
            print(f"  [wiki-{lang}-pages] 中断: {str(e)[:50]}", flush=True)
            break
        q = js.get("query", {})
        pages = q.get("pages", {})
        if not pages:
            break
        last_title = None
        for p in pages.values():
            ext, title = clean(p.get("extract") or ""), p.get("title") or ""
            last_title = title or last_title
            if len(ext) > 1200 and title not in seen:
                seen.add(title)
                docs.append(ext)
        if last_title:
            gapfrom = last_title
        print(f"  [wiki-{lang}-pages] {len(docs)}/{n} (next>={gapfrom[:18]})", flush=True)
        time.sleep(0.9)
    return docs[:n]


def split(docs: list[str], ratio: float) -> tuple[list[str], list[str]]:
    k = max(1, int(len(docs) * ratio))
    return docs[k:], docs[:k]


def dump(path: str, rows: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(r.replace("\n", " ").strip() + "\n")
    print(f"  wrote {os.path.basename(path)}: {len(rows)} 行 / {sum(len(r) for r in rows)/1e6:.2f} M 字符", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/data/cpt_domain")
    ap.add_argument("--arxiv-docs", type=int, default=2000, help="0 = 用缓存")
    ap.add_argument("--zh-domain", type=int, default=0, help="是否抓中文维基领域条目，0=跳过（默认单语实验）")
    ap.add_argument("--wiki-zh-gen", type=int, default=0)
    ap.add_argument("--wiki-en-gen", type=int, default=1500)
    ap.add_argument("--general-source", default="wikitext", choices=["wikitext", "wiki"])
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()
    cache = os.path.join(args.out, "_cache")
    os.makedirs(cache, exist_ok=True)

    # 1) 英文领域：arXiv cs.CR
    p = os.path.join(cache, f"arxiv_{args.arxiv_docs}.json")
    en = cache_load(p)
    if en:
        print(f"[1/5] arXiv 命中缓存: {len(en)} 篇", flush=True)
    else:
        print("[1/5] arXiv cs.CR …", flush=True)
        en = arxiv_docs(args.arxiv_docs)
        cache_save(p, en)
        print(f"  英文领域 {len(en)} 篇（已缓存）", flush=True)

    # 2) 中文领域：中文维基条目
    p = os.path.join(cache, "wiki_zh_domain.json")
    zh = cache_load(p)
    if args.zh_domain <= 0:
        zh = zh or []
        print(f"[2/5] 中文领域跳过（保留已有 {len(zh)} 篇不参与）", flush=True)
        zh = []
    elif zh:
        print(f"[2/5] 中文领域命中缓存: {len(zh)} 篇", flush=True)
    else:
        print("[2/5] 中文维基领域条目 …", flush=True)
        zh = wiki_titles("zh", ZH_TITLES)
        if len(zh) >= 20:
            cache_save(p, zh)
        print(f"  中文领域 {len(zh)} 篇", flush=True)

    # 3) 通用回放
    p = os.path.join(cache, f"wiki_zh_gen_{args.wiki_zh_gen}.json")
    rz = cache_load(p)
    if rz:
        print(f"[3/5] 中文通用命中缓存: {len(rz)} 篇", flush=True)
    else:
        print("[3/5] 中文通用语料（分页抽样）…", flush=True)
        rz = wiki_allpages("zh", args.wiki_zh_gen)
        cache_save(p, rz)
        print(f"  中文通用 {len(rz)} 篇", flush=True)

    p = os.path.join(cache, f"general_{args.general_source}_{args.wiki_en_gen}.json")
    re_ = cache_load(p)
    if re_:
        print(f"[3/5] 通用语料命中缓存: {len(re_)} 篇", flush=True)
    else:
        if args.general_source == "wikitext":
            print(f"[3/5] 通用语料 WikiText（{args.wiki_en_gen} 篇）…", flush=True)
            re_ = wikitext_docs(args.wiki_en_gen)
        else:
            print(f"[3/5] 通用语料 en-wiki 分页（{args.wiki_en_gen} 篇）…", flush=True)
            re_ = wiki_allpages("en", args.wiki_en_gen)
        cache_save(p, re_)
        print(f"  通用语料 {len(re_)} 篇（已缓存）", flush=True)

    print("[4/5] 切分落盘 …", flush=True)
    ztr, zva = split(zh, args.val_ratio) if zh else ([], [])
    etr, eva = split(en, args.val_ratio) if en else ([], [])
    rtr, rva = split(rz + re_, args.val_ratio)

    dump(os.path.join(args.out, "domain_train.txt"), ztr + etr)
    dump(os.path.join(args.out, "domain_val.txt"), zva + eva)
    dump(os.path.join(args.out, "replay_train.txt"), rtr)
    dump(os.path.join(args.out, "replay_val.txt"), rva)

    meta = {"domain": "arXiv cs.CR + zh.wikipedia(crypto/security/AI-safety)",
            "arxiv_docs": len(en), "zh_domain_docs": len(zh),
            "zh_general_docs": len(rz), "en_general_docs": len(re_),
            "domain_train_docs": len(ztr) + len(etr), "domain_val_docs": len(zva) + len(eva)}

    print("[5/5] token 规模 …", flush=True)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("D:/models/Qwen2.5-0.5B-Instruct")
        for name in ["domain_train", "domain_val", "replay_train", "replay_val"]:
            txt = open(os.path.join(args.out, f"{name}.txt"), encoding="utf-8").read()
            n = len(tok(txt, add_special_tokens=False)["input_ids"])
            meta[f"{name}_tokens"] = n
            print(f"  {name}: {n:,} tokens")
    except Exception as e:  # noqa: BLE001
        print(f"  tokenizer 跳过: {e}")

    with open(os.path.join(args.out, "corpus_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
