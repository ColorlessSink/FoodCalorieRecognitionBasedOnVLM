'''
从公开图片搜索接口抓取外部图片（50 类单食物 + 真实混合餐盘）
---
背景：主数据集 dataset_50cls 全部来自 ChineseFoodNet（同一采集源、风格高度一致），
混合餐盘更是从 test 集合成的（build_mixed_plates.py），存在两个已知分布盲区：
  ① 单食物图全是"数据集摄影风格"（正面、居中、白盘/黑底），
     换到真实手机随手拍场景表现如何从未验证；
  ② 混合餐盘是合成盘（圆贴圆、白盘暖灰桌），真实食堂/家常混合菜（大盘混装、
     菜贴菜边界不清）一图未测。
解决：从公开图片搜索抓一批网图建立 dataset_external/，作为"真实场景独立测试集"
接入评估流程。它与 dataset_50cls 完全隔离（独立目录、独立标签 CSV、独立评估脚本），
**不触碰任何主数据集文件，不影响既有流程**

数据来源（都是浏览器可达的公开搜索接口，无登录、无付费）：
  ① 百度图片 acjson（image.baidu.com/search/acjson）——中文菜名主力源；
  ② Bing 图片 async（www.bing.com/images/async）——补充源，murl 在 iusc JSON 里。
  每类查 2 个词形（"{菜名} 菜品" / "{菜名} 家常做法"），两源各取一部分，双源去重

工程要点（都有实测依据）：
  - 可续跑：进度实时写 scrape_progress.json，中断后重跑只补缺类，已抓的类跳过；
  - md5 去重：同图多源/多词形命中同一图很常见，按内容 md5 全局去重；
  - 尺寸过滤：<200px 短边丢弃（识别器会缩到 224，太小的图多为缩略图/表情包）；
  - 中文路径：Windows 下 cv2.imwrite 对中文目录写失败 → 用 imencode + tofile；
  - 失败容忍：单图下载失败/超时/解码失败只计数不中断（网图 10~20% 失联是常态）；
  - 出处留痕：manifest CSV 记录每张图的 url/来源/查询词/原始宽高，报告可交代数据来源

输出（全部在 dataset_external/ 下，与主数据集隔离）：
  dataset_external/images/<类名>/xxx.jpg          每类约 N_PER_CLASS 张单食物图
  dataset_external/mixed_raw/xxx.jpg              真实混合菜图（另脚本标注）
  dataset_external/scrape_manifest.csv            所有成功图片的出处清单
  dataset_external/scrape_progress.json           断点续跑进度
'''
import os, sys, io, csv, json, time, hashlib, random, re

import requests
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import ROOT, load_classes

# ---------------- 可调参数 ----------------
N_PER_CLASS = 10          # 每类目标抓取张数（两源合计）
N_MIXED = 40              # 真实混合菜图目标张数
TIMEOUT = 12              # 单图下载超时（秒）
MIN_SIDE = 200            # 短边下限（识别器输入 224px，太小多为缩略图）
MAX_PER_QUERY_PAGE = 60  # 单词形单源最多取多少候选 URL（rn/first 上限内）

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://image.baidu.com/",
}

# 混合菜搜索词（真实餐桌/食堂场景，不是合成盘；覆盖 2~4 菜混装 + 餐盘分格）
MIXED_QUERIES = [
    "食堂两菜一饭 餐盘", "食堂饭菜 餐盘", "快餐两荤一素",
    "家常菜 三个菜 摆盘", "一桌家常菜", "米饭炒菜 餐盘",
    "盒饭 三个菜", "盖浇饭 双拼", "大排档 拼盘 菜",
]
# 单食物图排除词（命中标题则跳过该图，避免抓到菜谱步骤图/插画）
BAD_TITLE_WORDS = ["做法", "步骤", "教程", "图解", "插画", "手绘", "卡通", "漫画",
                   "logo", "标志", "菜单", "价目", "名片", "海报", "banner", "icon"]


# ---------------- 搜索接口适配 ----------------

def search_baidu(query, n_want, session):
    # 百度图片 acjson 接口 → [(url, w, h, title)]。实测无需登录，直接 JSON
    out = []
    pn = 0
    while len(out) < n_want and pn < 120:
        params = {
            "tn": "resultjson_com", "ipn": "rj", "word": query,
            "pn": pn, "rn": 30, "ie": "utf-8", "oe": "utf-8",
            "queryType": "0", "istype": "2", "face": "0",
        }
        try:
            r = session.get("https://image.baidu.com/search/acjson",
                            params=params, headers=UA, timeout=10)
            data = r.json().get("data", [])
        except Exception:
            # 实测：acjson 偶发返回带非法转义的 JSON（下载页源码混入），
            # requests 的 r.json() 直接抛异常 → 整页候选全丢。降级用正则
            # 从原始文本里抠 thumbURL，比整页丢弃好。
            data = []
            for m in re.finditer(r'"(thumb|middle)URL":"([^"]+)"', r.text):
                data.append({"middleURL": m.group(2).replace("\\/", "/")})
        if not data:
            break
        for d in data:
            if not d:
                continue
            url = d.get("middleURL") or d.get("thumbURL") or d.get("objURL")
            if not url:
                continue
            w = d.get("width") or 0
            h = d.get("height") or 0
            out.append((url, w, h, d.get("fromPageTitleEnc") or ""))
        pn += 30
        time.sleep(0.3)   # 接口礼貌间隔
    return out[:n_want]


def search_bing(query, n_want, session):
    # Bing 图片 async 接口 → [(url, w, h, title)]
    # 实测 murl 藏在 class="iusc" 的 JSON 属性里（HTML 实体编码），需反转义后解析
    out = []
    first = 0
    while len(out) < n_want and first < 150:
        params = {"q": query, "first": first, "count": "35", "mmasync": "1"}
        try:
            r = session.get("https://www.bing.com/images/async",
                            params=params, headers=UA, timeout=10)
            t = r.text
        except Exception:
            break
        items = re.findall(r'class="iusc"[^>]*m="([^"]+)"', t)
        if not items:
            break
        for blob in items:
            try:
                j = json.loads(blob.replace("&quot;", '"')
                               .replace("&amp;", "&").replace("&#39;", "'"))
            except Exception:
                continue
            url = j.get("murl")
            if not url:
                continue
            # 宽高不在 m 里，用 turl 的 OIP 参数兜底不可靠，先记 0（下载后实测）
            out.append((url, 0, 0, j.get("t") or ""))
        first += 35
        time.sleep(0.3)
    return out[:n_want]


# ---------------- 下载与校验 ----------------

def download_image(url, session):
    # 下载单图 → BGR ndarray 或 None（失败/非图片/太小）
    try:
        r = session.get(url, headers=UA, timeout=TIMEOUT)
        if r.status_code != 200 or len(r.content) < 4000:   # <4KB 基本是占位图
            return None
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIDE:
            return None
        # 灰度近全黑/全白的死图过滤
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if g.std() < 10:
            return None
        return img
    except Exception:
        return None


def content_md5(img):
    return hashlib.md5(img.tobytes()).hexdigest()


def save_unicode(path, img):
    # Windows 中文路径：cv2.imwrite 失败 → imencode + tofile
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(path)
    return ok


def load_manifest(path):
    # 读已有 manifest（续跑时跳过已抓图片）
    rows, seen_md5 = [], set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                if row.get("md5"):
                    seen_md5.add(row["md5"])
    return rows, seen_md5


def append_manifest(path, row):
    # manifest 追加一行（首行写表头，续跑时表头已存在）
    header = ["file", "md5", "source", "query", "url", "w", "h", "title"]
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        wtr = csv.writer(f)
        if new_file:
            wtr.writerow(header)
        wtr.writerow([row[k] for k in header])


def title_ok(title):
    # 排除菜谱步骤图/插画/图标类标题（中英文关键词都查）
    tl = title.lower()
    return not any(b in tl for b in BAD_TITLE_WORDS)


# ---------------- 主流程 ----------------

def scrape_single_food(session, names_zh, out_root, progress, manifest_path):
    # 每类抓 N_PER_CLASS 张：百度为主（6）、Bing 补（4）
    img_root = os.path.join(out_root, "images")
    n_cls_done = progress.get("single_done_classes", [])
    total_saved, seen_md5 = progress.get("single_total_saved", 0), None
    _, seen_md5 = load_manifest(manifest_path)

    for idx, zh in names_zh:
        if zh in n_cls_done:
            continue
        cls_dir = os.path.join(img_root, zh)
        os.makedirs(cls_dir, exist_ok=True)
        saved = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")])

        # 已够数的类直接标完成（部分成功后重跑的场景）
        if saved >= N_PER_CLASS:
            n_cls_done.append(zh)
            progress["single_done_classes"] = n_cls_done
            _dump_progress(progress)
            continue

        queries = [f"{zh} 菜品", f"{zh} 家常做法"]
        plan = [(search_baidu, queries[0], 6), (search_bing, queries[1], 4)]
        tried = 0
        for fn, query, quota in plan:
            if saved >= N_PER_CLASS:
                break
            cands = fn(query, quota * 3, session)   # 多取 3 倍候选，下载失败有冗余
            for url, _w, _h, title in cands:
                if saved >= N_PER_CLASS or tried > 40:
                    break
                if not title_ok(title):
                    continue
                tried += 1
                img = download_image(url, session)
                if img is None:
                    continue
                md5 = content_md5(img)
                if md5 in seen_md5:
                    continue
                seen_md5.add(md5)
                h, w = img.shape[:2]
                fname = f"{zh}_{saved + 1:03d}.jpg"
                if not save_unicode(os.path.join(cls_dir, fname), img):
                    continue
                saved += 1
                total_saved += 1
                append_manifest(manifest_path, {
                    "file": f"images/{zh}/{fname}", "md5": md5, "source": fn.__name__,
                    "query": query, "url": url, "w": w, "h": h, "title": title,
                })
        status = "OK" if saved >= N_PER_CLASS else f"仅{saved}张"
        print(f"  [{idx + 1:2d}/50] {zh}: {saved}/{N_PER_CLASS} {status}")
        n_cls_done.append(zh)
        progress["single_done_classes"] = n_cls_done
        progress["single_total_saved"] = total_saved
        _dump_progress(progress)
        time.sleep(random.uniform(0.5, 1.2))   # 类间随机间隔，降低封禁风险

    return total_saved


def scrape_mixed(session, out_root, progress, manifest_path):
    # 真实混合菜图：多词形轮询，每词形出几图，避免同质化
    mixed_dir = os.path.join(out_root, "mixed_raw")
    os.makedirs(mixed_dir, exist_ok=True)
    _, seen_md5 = load_manifest(manifest_path)
    saved = progress.get("mixed_saved", 0)
    existing = {f for f in os.listdir(mixed_dir) if f.endswith(".jpg")}
    saved = max(saved, len(existing))

    per_query = max(2, N_MIXED // len(MIXED_QUERIES))
    for qi, query in enumerate(MIXED_QUERIES):
        cands = search_baidu(query, per_query * 3, session)
        got = 0
        for url, _w, _h, title in cands:
            if got >= per_query or saved >= N_MIXED:
                break
            if not title_ok(title):
                continue
            img = download_image(url, session)
            if img is None:
                continue
            md5 = content_md5(img)
            if md5 in seen_md5:
                continue
            seen_md5.add(md5)
            h, w = img.shape[:2]
            fname = f"mixed_{saved + 1:03d}.jpg"
            if not save_unicode(os.path.join(mixed_dir, fname), img):
                continue
            saved += 1
            got += 1
            append_manifest(manifest_path, {
                "file": f"mixed_raw/{fname}", "md5": md5, "source": "search_baidu",
                "query": query, "url": url, "w": w, "h": h, "title": title,
            })
        print(f"  [混合 {qi + 1}/{len(MIXED_QUERIES)}] {query}: +{got}（累计 {saved}/{N_MIXED}）")
        progress["mixed_saved"] = saved
        _dump_progress(progress)
        time.sleep(random.uniform(0.5, 1.2))

    return saved


def _dump_progress(progress, path=None):
    path = path or os.path.join(ROOT, "dataset_external", "scrape_progress.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out_root = os.path.join(ROOT, "dataset_external")
    os.makedirs(out_root, exist_ok=True)
    manifest_path = os.path.join(out_root, "scrape_manifest.csv")
    prog_path = os.path.join(out_root, "scrape_progress.json")

    progress = {"single_done_classes": [], "single_total_saved": 0, "mixed_saved": 0}
    if os.path.exists(prog_path):
        with open(prog_path, "r", encoding="utf-8") as f:
            progress.update(json.load(f))

    names_zh, _idxs, _df = load_classes(os.path.join(ROOT, "dataset_50cls"))
    session = requests.Session()

    print(f"[1/2] 单食物：50 类 × {N_PER_CLASS} 张 → dataset_external/images/")
    n_single = scrape_single_food(session, list(enumerate(names_zh)), out_root, progress, manifest_path)
    print(f"  单食物累计入库 {n_single} 张")

    print(f"[2/2] 真实混合菜：{N_MIXED} 张 → dataset_external/mixed_raw/")
    n_mixed = scrape_mixed(session, out_root, progress, manifest_path)
    print(f"  混合菜累计入库 {n_mixed} 张")

    # 汇总
    print("\n==== 抓取汇总 ====")
    for sub in ["images", "mixed_raw"]:
        d = os.path.join(out_root, sub)
        if os.path.isdir(d):
            cnt = sum(len([x for x in fs if x.endswith(".jpg")])
                      for _r, _d, fs in os.walk(d))
            print(f"  {sub}: {cnt} 张")
    print(f"  manifest: {len(load_manifest(manifest_path)[0])} 行")


if __name__ == "__main__":
    main()
