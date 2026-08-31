'''
测试集三场景【人工】标注工具
---
背景：data/build_scene_split.py 用亮度/饱和度/清晰度/长宽比四个统计量加权打分，
     每类强制 4/4/4 三分——统计量与"人眼觉得难不难认"只有弱相关，划分不准，
     故改成本工具人工看图标注

用法：
    python tools/label_scene.py        # 在任意目录执行均可，路径由 __file__ 推算

按键：
    1 / 2 / 3    标为 standard / real / challenge，自动跳下一张
    ← / →        前后翻页（只浏览，不改标注）
    BackSpace    清除当前图的标注
    n            跳到下一张未标注的图
    s            开/关「50类 × 3场景」计数窗口（标注时尽量往均衡靠）
    e            导出最终 CSV（要求 600 张全部标完）
    Esc          退出（进度随标随存，可随时中断续标）

与 build_scene_split.py 的输入/输出对齐：
    输入  dataset_50cls/test.csv（split,path,label）
    输出  dataset_50cls/test_scene.csv：列 split,path,label,scene，
          scene ∈ {standard,real,challenge}，utf-8-sig 编码，行序与 test.csv 一致
    消费端 experiments/baseline_eval.py 按 path（对 \\ / 归一）匹配 scene，完全兼容

与自动版的区分：
    · 标注进度实时写入 dataset_50cls/scene_labels_progress.json，可中断续标；
    · 首次导出前把旧的自动版备份为 dataset_50cls/test_scene_auto.csv；
    · 导出时打印人工 vs 自动的一致率与 Cohen's kappa（可写进报告当消融叙事）
'''
import json, os, shutil
from collections import Counter

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset_50cls")
TEST_CSV = os.path.join(DATA, "test.csv")
OUT_CSV = os.path.join(DATA, "test_scene.csv")
AUTO_BAK = os.path.join(DATA, "test_scene_auto.csv")
PROGRESS_JSON = os.path.join(DATA, "scene_labels_progress.json")

SCENES = ("standard", "real", "challenge")
SCENE_ZH = {"standard": "①标准 standard", "real": "②真实 real", "challenge": "③困难 challenge"}
SCENE_COLOR = {"standard": "#1a7f37", "real": "#b25e09", "challenge": "#c62828"}
IMG_BOX = (840, 620)
KEY2SCENE = {"1": "standard", "2": "real", "3": "challenge",
             "KP_1": "standard", "KP_2": "real", "KP_3": "challenge"}  # 主键盘+小键盘

GUIDE = """\
【先问：食物主体是否一眼可辨、占画面主要比例？】
不满足，或需要费力寻找 → 直接判 challenge。
主体清晰的，再看环境真实感分 standard / real。

① standard 标准/理想场景（键 1）
  摆拍图：食物居中、占画面大；背景纯色/干净
  或完全虚化；光线充足均匀、色彩鲜亮；对焦
  清晰；无遮挡、无杂物。
  典型：菜谱网站、美食图库图。

② real 真实日常场景（键 2）
  真实就餐环境：家庭餐桌、餐厅、外卖盒/
  打包袋；自然光或室内灯光，允许轻微阴影；
  画面中出现餐具、桌布、手、相邻菜品等
  日常元素；但食物主体仍清晰、占比够大。
  典型：手机随手拍的吃饭照。

③ challenge 困难场景（键 3，命中任一条即可）
  · 光线昏暗 / 过曝 / 明显色偏
  · 模糊失焦或运动抖动
  · 食物过小过远、只拍到一角、被遮挡
  · 俯拍满桌多菜，主体不突出
  · 强反光（油光/塑料盒/玻璃）
  · 背景杂乱到主体被淹没
"""


class Labeler:
    def __init__(self, root):
        self.root = root
        self.df = pd.read_csv(TEST_CSV)
        self.labels = self._load_progress()
        # label -> 中文类名（取 path 的上级目录名，如 00_麻婆豆腐）
        self.cls_name = {}
        for _, r in self.df.iterrows():
            p = str(r["path"]).replace("\\", "/")
            self.cls_name.setdefault(r["label"], os.path.basename(os.path.dirname(p)))
        self.i = self._first_unlabeled()
        self._photo = None      # 挂实例属性：局部变量的 PhotoImage 会被 GC，图片就不显示了
        self.stats_win = None
        self.stats_text = None
        self._build_ui()
        self._bind_keys()
        self.show()

    # ---------- 路径/进度 ----------
    def _path(self, i):
        return str(self.df.iloc[i]["path"])

    def _load_progress(self):
        if not os.path.exists(PROGRESS_JSON):
            return {}
        with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        valid = set(self.df["path"].astype(str))
        d = {k: v for k, v in d.items() if k in valid and v in SCENES}
        print(f"载入 {len(d)} 条已标注记录")
        return d

    def _save_progress(self):
        with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=1)

    def _first_unlabeled(self):
        for idx in range(len(self.df)):
            if self._path(idx) not in self.labels:
                return idx
        return 0

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title("三场景人工标注 · test_scene.csv（替代 build_scene_split.py 自动划分）")
        self.top = tk.Label(self.root, font=("Microsoft YaHei UI", 13, "bold"),
                            anchor="w", pady=6)
        self.top.pack(fill="x", padx=10)

        mid = tk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=10, pady=4)

        box = tk.Frame(mid, width=IMG_BOX[0], height=IMG_BOX[1], bg="#1e1e1e")
        box.pack(side="left", fill="both", expand=True)
        box.pack_propagate(False)   # 固定图区大小：坏图时布局不塌缩
        self.img_label = tk.Label(box, bg="#1e1e1e", fg="#eee",
                                  font=("Microsoft YaHei UI", 11))
        self.img_label.pack(fill="both", expand=True)

        guide = tk.Text(mid, width=68, wrap="word", bg="#fafafa",
                        relief="groove", padx=8, pady=8,
                        font=("Microsoft YaHei UI", 10))
        guide.insert("1.0", GUIDE)
        guide.configure(state="disabled")
        guide.pack(side="right", fill="y")

        self.count_label = tk.Label(self.root, anchor="w",
                                     font=("Microsoft YaHei UI", 11))
        self.count_label.pack(fill="x", padx=10, pady=2)

        keys = ("快捷键:  [1]标准  [2]真实  [3]困难   |   [←/→]翻页  "
                "[Backspace]清除当前标注  [n]下一张未标  [s]统计  [e]导出  [Esc]退出")
        tk.Label(self.root, text=keys, anchor="w", fg="#555",
                 font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=10, pady=(0, 8))

    def _bind_keys(self):
        for k, sc in KEY2SCENE.items():
            self.root.bind_all(f"<Key-{k}>", lambda e, sc=sc: self.assign(sc))
        self.root.bind_all("<Left>", lambda e: self.move(-1))
        self.root.bind_all("<Right>", lambda e: self.move(1))
        self.root.bind_all("<BackSpace>", lambda e: self.clear_current())
        self.root.bind_all("<Key-n>", lambda e: self.jump_unlabeled())
        self.root.bind_all("<Key-s>", lambda e: self.toggle_stats())
        self.root.bind_all("<Key-e>", lambda e: self.export())
        self.root.bind_all("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- 渲染 ----------
    def show(self):
        row = self.df.iloc[self.i]
        path = self._path(self.i)
        sc = self.labels.get(path)
        self.top.config(
            text=f"[{self.i + 1}/{len(self.df)}]  {self.cls_name[row['label']]}    "
                 f"当前: {SCENE_ZH[sc] if sc else '未标注'}    "
                 f"总进度 {len(self.labels)}/{len(self.df)}",
            fg=SCENE_COLOR.get(sc, "#333"))
        self._render_image(path)
        self._update_counts()
        if self.stats_win is not None and self.stats_win.winfo_exists():
            self._refresh_stats()

    def _render_image(self, path):
        fp = os.path.join(DATA, path.replace("\\", "/"))
        try:
            img = Image.open(fp)
            img.load()                 # 立即解码并释放文件句柄
            img = img.convert("RGB")
            img.thumbnail(IMG_BOX)    # 只缩不放，保持纵横比
            self._photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=self._photo, text="")
        except Exception as exc:
            self._photo = None
            self.img_label.configure(image="", text=f"无法打开图片:\n{fp}\n\n{exc}")

    def _update_counts(self):
        cl = self.df.iloc[self.i]["label"]
        total, cur = Counter(), Counter()
        for _, r in self.df.iterrows():
            s = self.labels.get(str(r["path"]))
            if not s:
                continue
            total[s] += 1
            if r["label"] == cl:
                cur[s] += 1
        self.count_label.config(
            text=f"整体  标准 {total['standard']} / 真实 {total['real']} / "
                 f"困难 {total['challenge']}      本类({self.cls_name[cl]})  "
                 f"{cur['standard']} / {cur['real']} / {cur['challenge']}")

    # ---------- 动作 ----------
    def assign(self, scene):
        self.labels[self._path(self.i)] = scene
        self._save_progress()          # 每标一张即落盘，可随时中断
        self.move(1)

    def move(self, delta):
        self.i = (self.i + delta) % len(self.df)
        self.show()

    def clear_current(self):
        path = self._path(self.i)
        if path in self.labels:
            del self.labels[path]
            self._save_progress()
        self.show()

    def jump_unlabeled(self):
        n = len(self.df)
        for step in range(1, n + 1):
            j = (self.i + step) % n
            if self._path(j) not in self.labels:
                self.i = j
                self.show()
                return
        self.show()
        messagebox.showinfo("提示", "600 张已全部标注完成，可按 [e] 导出。")

    # ---------- 统计窗口（s 键开关） ----------
    def toggle_stats(self):
        if self.stats_win is not None and self.stats_win.winfo_exists():
            self.stats_win.destroy()
            self.stats_win = None
            return
        self.stats_win = tk.Toplevel(self.root)
        self.stats_win.title("50类 × 3场景 计数（标注时尽量别让某场景长期为 0）")
        self.stats_text = tk.Text(self.stats_win, width=60, font=("Consolas", 10))
        self.stats_text.pack(fill="both", expand=True)
        self._refresh_stats()

    def _refresh_stats(self):
        per = {lab: Counter() for lab in self.cls_name}
        for _, r in self.df.iterrows():
            s = self.labels.get(str(r["path"]))
            if s:
                per[r["label"]][s] += 1
        lines = ["类别              std  real  chl   已/总", "-" * 46]
        for lab in self.cls_name:
            c = per[lab]
            n_cls = int((self.df["label"] == lab).sum())
            lines.append(f"{self.cls_name[lab]:<16} {c['standard']:>3}  {c['real']:>3}  "
                         f"{c['challenge']:>3}   {sum(c.values())}/{n_cls}")
        tot = Counter()
        for c in per.values():
            tot.update(c)
        lines.append("-" * 46)
        lines.append(f"{'合计':<16} {tot['standard']:>3}  {tot['real']:>3}  "
                     f"{tot['challenge']:>3}   {sum(tot.values())}/{len(self.df)}")
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.configure(state="disabled")

    # ---------- 导出 ----------
    def export(self):
        n, done = len(self.df), len(self.labels)
        if done < n:
            messagebox.showwarning(
                "未完成",
                f"还有 {n - done} 张未标注（已标 {done}/{n}）。\n"
                "为避免下游 baseline_eval 静默丢样本，标完再导出。")
            return

        bak_info = ""
        if os.path.exists(OUT_CSV):
            if not os.path.exists(AUTO_BAK):
                shutil.copyfile(OUT_CSV, AUTO_BAK)
                bak_info = f"旧自动版已备份为 {os.path.basename(AUTO_BAK)}"
            else:
                bak_info = f"备份已存在: {os.path.basename(AUTO_BAK)}（未覆盖）"

        out = self.df.copy()
        out["scene"] = out["path"].astype(str).map(self.labels)
        out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

        dist = Counter(out["scene"])
        print("场景分布:", dict(dist))
        print(bak_info)
        print("已写入", OUT_CSV)

        # 人工 vs 自动的一致率 + Cohen's kappa（报告里的消融叙事素材）
        agree_info = ""
        if os.path.exists(AUTO_BAK):
            try:
                old = pd.read_csv(AUTO_BAK)
                old_map = dict(zip(old["path"].astype(str), old["scene"]))
                auto = out["path"].astype(str).map(old_map)
                both = out["scene"].notna() & auto.notna()
                po = float((out["scene"][both] == auto[both]).mean())
                hum_p = out["scene"].value_counts(normalize=True)
                aut_p = auto[both].value_counts(normalize=True)
                pe = float(sum(hum_p.get(c, 0.0) * aut_p.get(c, 0.0) for c in SCENES))
                kappa = (po - pe) / (1 - pe + 1e-9)
                agree_info = f"\n与自动版一致率 {po:.3f}，Cohen's kappa {kappa:.3f}"
                print(f"与自动版一致率 {po:.3f}，kappa {kappa:.3f}")
            except Exception as exc:
                print("对比计算失败:", exc)

        messagebox.showinfo(
            "导出成功",
            f"已写入 {os.path.basename(OUT_CSV)}\n{bak_info}\n\n"
            f"标准 {dist['standard']} / 真实 {dist['real']} / 困难 {dist['challenge']}"
            f"{agree_info}")

    def on_close(self):
        n, done = len(self.df), len(self.labels)
        if done < n:
            if not messagebox.askyesno(
                    "退出",
                    f"还有 {n - done} 张未标注。\n进度已保存在:\n{PROGRESS_JSON}\n确定退出？"):
                return
        self._save_progress()
        self.root.destroy()


def main():
    root = tk.Tk()
    Labeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()
