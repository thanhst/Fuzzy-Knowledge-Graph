"""
run_all.py — Chạy TOÀN BỘ pipeline thực nghiệm theo đúng thứ tự:
  KB1 -> KB2 -> KB3 -> KB4 -> KB5 -> KB6 -> Ablation -> Baseline -> Báo cáo

Cách dùng:
  python3 run_all.py                 # chạy đầy đủ (có thể mất 30-90 phút
                                        tuỳ quy mô dữ liệu thật, xem README)
  python3 run_all.py --quick          # chạy nhanh (epochs/n_seeds giảm) để
                                        kiểm tra toàn bộ pipeline không lỗi
                                        trước khi chạy đầy đủ trên dữ liệu thật
  python3 run_all.py --only kb1,kb6   # chỉ chạy các KB được liệt kê
"""
import sys, os, argparse, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Chạy nhanh với epochs/n_seeds giảm để kiểm tra pipeline")
    parser.add_argument("--only", type=str, default="",
                        help="Danh sách KB cách nhau bởi dấu phẩy, ví dụ: kb1,kb3,kb6")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="Cho phép dữ liệu synthetic khi thiếu dữ liệu thật (không dùng báo cáo)")
    parser.add_argument("--brset-only", action="store_true",
                        help="Chỉ chạy trên BRSET thật; bỏ hai bộ dữ liệu thô còn thiếu ở KB1")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Ghi đè số epoch huấn luyện FKG-E (mặc định: 3 nếu --quick, 100 nếu đầy đủ)")
    parser.add_argument("--run-id", type=str, default="",
                        help="Mã lần chạy; mặc định sinh theo thời gian")
    args = parser.parse_args()

    if args.quick:
        print(">>> CHẾ ĐỘ NHANH: giảm epochs=3, n_seeds=1, thu nhỏ lưới quét để "
              "kiểm tra pipeline không lỗi (không dùng để báo cáo chính thức).\n")
        C.FKGE.epochs = 3
        C.EVAL.N_SEEDS = 1
        C.EVAL.QUICK = True
        # Thu nhỏ mạnh các lưới quét KB3/KB4/KB5/KB6 -- mục đích --quick chỉ là
        # xác nhận KHÔNG LỖI RUNTIME, không phải chạy đủ độ phân giải thật.
        C.KB3_DIMS = [8, 32]
        C.KB4_LAMBDA_GRID = [0.3, 1.0]
        C.KB4_BETA_GRID = [0.3, 1.0]
        C.KB4_MULTIPLIERS = [0.0, 1.0]
        C.KB5_W_GRID = [1, 2]
        C.KB5_K_GRID = [2, 5]
        C.KB6_SAMPLE_RATIOS = [0.4, 1.0]

    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs phải >= 1")
        C.FKGE.epochs = args.epochs
        print(f">>> FKG-E epochs={C.FKGE.epochs}")

    only = set(x.strip().lower() for x in args.only.split(",") if x.strip()) or None

    if args.brset_only:
        from data.frb_package import package_available
        if not package_available(C.PATHS.BRSET_FRB_PACKAGE):
            raise SystemExit("Không tìm thấy gói FRB BRSET thật; dừng để tránh dùng dữ liệu synthetic.")

    if ((only is None or "kb1" in only)
            and not args.quick and not args.allow_synthetic and not args.brset_only
            and (not os.path.exists(C.PATHS.DIABETES_KAGGLE_RAW_FILE)
                 or not os.path.exists(C.PATHS.HEALTHCARE_DIABETES_RAW_FILE))):
        raise SystemExit(
            "KB1 thiếu dữ liệu thật Diabetes/Healthcare. Dùng --quick hoặc "
            "--allow-synthetic chỉ để kiểm tra pipeline; không dùng kết quả đó cho luận án."
        )

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    C.PATHS.OUTPUT_DIR = os.path.join(C.PATHS.ROOT, "outputs", "runs", run_id)

    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    t_start = time.time()
    requested = sorted(only) if only is not None else [
        "kb1", "kb2", "kb3", "kb4", "kb5", "kb6", "ablation", "baseline"
    ]
    from report.run_manifest import create_manifest, finish_manifest, write_manifest
    manifest = create_manifest(run_id, args.quick, requested, sys.argv)
    write_manifest(manifest)

    if only is None or "kb1" in only or "kb2" in only:
        from experiments.kb1_kb2 import run_kb1, run_kb2
        import json
        if only is None or "kb1" in only:
            r1 = run_kb1(brset_only=args.brset_only)
            json.dump(r1, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb1_results.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
        if only is None or "kb2" in only:
            r2, verdict = run_kb2()
            json.dump({"rows": r2, "verdict": verdict},
                      open(os.path.join(C.PATHS.OUTPUT_DIR, "kb2_results.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb3" in only:
        from experiments.kb3_kb5 import run_kb3
        import json
        r3 = run_kb3()
        json.dump(r3, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb3_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb5" in only:
        from experiments.kb3_kb5 import run_kb5
        import json
        n_seeds_kb5 = 1 if args.quick else C.EVAL.N_SEEDS
        r5 = run_kb5(n_seeds=n_seeds_kb5)
        json.dump(r5, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb5_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb4" in only:
        from experiments.kb4_kb6 import run_kb4
        import json
        n_seeds_kb4 = 1 if args.quick else C.EVAL.N_SEEDS
        r4 = run_kb4(n_seeds=n_seeds_kb4)
        json.dump(r4, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb4_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb6" in only:
        from experiments.kb4_kb6 import run_kb6
        import json
        n_seeds_kb6 = 1 if args.quick else C.EVAL.N_SEEDS
        r6 = run_kb6(n_seeds=n_seeds_kb6)
        json.dump(r6, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb6_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "ablation" in only:
        from experiments.ablation import run_ablation
        import json
        rows = run_ablation()
        json.dump(rows, open(os.path.join(C.PATHS.OUTPUT_DIR, "ablation_results.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "baseline" in only:
        from experiments.baseline_comparison import run_baseline_comparison
        import json
        rows = run_baseline_comparison()
        json.dump(rows, open(os.path.join(C.PATHS.OUTPUT_DIR, "baseline_comparison.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    finish_manifest(manifest, elapsed)
    write_manifest(manifest)
    print(f"\n{'='*70}\nHOÀN TẤT TOÀN BỘ THỰC NGHIỆM trong {elapsed/60:.1f} phút.\n{'='*70}")

    print("\nĐang sinh báo cáo tổng hợp (bảng + biểu đồ)...")
    from report.generate_report import main as gen_report
    gen_report()


if __name__ == "__main__":
    main()
