#!/usr/bin/env python3
"""
CUDA 练习题自动验证脚本

用法:
  python check_exercises.py                # 验证所有练习
  python check_exercises.py 01_vector_add   # 只验证某个模块
  python check_exercises.py --list          # 列出所有可验证的练习
"""

import os, sys, subprocess, glob, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

EXERCISES = {
    "01_vector_add": {
        "files": ["ex1_saxpy_level1", "ex1_saxpy_level2",
                  "ex2_relu_level1", "ex2_relu_level2",
                  "ex3_fma_level1", "ex3_fma_level2"],
        "dir": "01_vector_add/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS"],
    },
    "02_matrix_mul": {
        "files": ["ex1_transpose_level1", "ex1_transpose_level2",
                  "ex2_matadd_level1", "ex2_matadd_level2",
                  "ex3_gemv_tiled_level1", "ex3_gemv_tiled_level2"],
        "dir": "02_matrix_mul/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS"],
    },
    "03_reduce": {
        "files": ["ex1_reduce_max_level1", "ex1_reduce_max_level2",
                  "ex2_dot_level1", "ex2_dot_level2",
                  "ex3_count_level1", "ex3_count_level2"],
        "dir": "03_reduce/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS"],
    },
    "05_bank_conflict": {
        "files": ["ex1_padding_level1", "ex1_padding_level2",
                  "ex2_transpose_smem_level1", "ex2_transpose_smem_level2"],
        "dir": "05_bank_conflict/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS", "带宽"],
    },
    "06_coalescing": {
        "files": ["ex1_aos_soa_level1", "ex1_aos_soa_level2",
                  "ex2_write_pattern_level1", "ex2_write_pattern_level2"],
        "dir": "06_coalescing/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS"],
    },
    "07_softmax": {
        "files": ["ex1_logsumexp_level1", "ex2_l2norm_level1",
                  "ex3_softmax_v1_level2"],
        "dir": "07_softmax/exercises",
        "expected_output": ["✓", "通过", "正确", "PASS"],
    },
    "debug_exercises": {
        "files": ["bug1_vector_add", "bug2_reduce", "bug3_softmax"],
        "dir": "debug_exercises",
        "expected_output": None,  # Bug exercises don't pass until fixed
        "is_bug": True,
    },
}


def get_gpu_arch():
    """Auto-detect GPU architecture."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        cap = result.stdout.strip().replace(".", "")
        if cap:
            return f"sm_{cap}"
    except Exception:
        pass
    return "sm_70"  # default fallback


def compile_exercise(exercise_path, arch):
    """Compile a CUDA exercise file."""
    cmd = ["nvcc", "-O2", f"-arch={arch}", "-o",
           str(exercise_path.with_suffix("")), str(exercise_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        # Return error but don't fail entirely (some exercises may need special setup)
        return False, result.stderr.strip()
    return True, None


def run_exercise(exercise_path, timeout=10):
    """Run a compiled exercise and check output."""
    binary = str(exercise_path.with_suffix(""))
    try:
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "(timeout)"
    except Exception as e:
        return -1, str(e)


def check_output(output, expected_patterns, is_bug=False):
    """Check if output contains expected success indicators."""
    if is_bug:
        # Bug exercises: successful if they compile (they're knowingly buggy)
        return None  # neutral - can't auto-verify
    if not expected_patterns:
        return True  # no specific check
    for pattern in expected_patterns:
        if pattern in output:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="CUDA 练习题自动验证")
    parser.add_argument("module", nargs="?", help="只验证指定模块")
    parser.add_argument("--list", action="store_true", help="列出所有可验证的练习")
    parser.add_argument("--arch", help="手动指定 GPU 架构 (如 sm_80)")
    parser.add_argument("--timeout", type=int, default=10, help="每个练习的超时时间(秒)")
    args = parser.parse_args()

    if args.list:
        print("可验证的练习模块:")
        for name, info in EXERCISES.items():
            count = len(info["files"])
            bug_tag = " [有 bug, 需手动修复]" if info.get("is_bug") else ""
            print(f"  {name}: {count} 个文件{bug_tag}")
        return

    arch = args.arch or get_gpu_arch()
    print(f"GPU 架构: {arch}\n")

    modules_to_check = [args.module] if args.module else list(EXERCISES.keys())

    total_ok = 0
    total_fail = 0
    total_skip = 0

    for module in modules_to_check:
        if module not in EXERCISES:
            print(f"未知模块: {module}")
            continue

        info = EXERCISES[module]
        exercise_dir = PROJECT_ROOT / info["dir"]
        is_bug = info.get("is_bug", False)

        print(f"{'='*60}")
        print(f"模块: {module} ({'debug 练习' if is_bug else 'Level 1/2 练习'})")
        print(f"{'='*60}")

        for fname in info["files"]:
            src = exercise_dir / f"{fname}.cu"
            if not src.exists():
                print(f"  [{fname}] 文件不存在, 跳过")
                total_skip += 1
                continue

            # Compile
            ok, err = compile_exercise(src, arch)
            if not ok:
                print(f"  [{fname}] 编译失败: {err[:100]}")
                total_fail += 1
                continue

            # Run
            retcode, output = run_exercise(src, args.timeout)

            if is_bug:
                print(f"  [{fname}] 编译+运行成功 (bug 练习, 需手动检查)")
                total_skip += 1
            elif retcode == 0:
                passed = check_output(output, info["expected_output"])
                if passed:
                    print(f"  [{fname}] ✓ 通过")
                    total_ok += 1
                elif passed is None:
                    print(f"  [{fname}] ? 无法自动判断, 请手动检查")
                    total_skip += 1
                else:
                    # Show first line of output for debugging
                    first_line = output.strip().split('\n')[0] if output else "(no output)"
                    print(f"  [{fname}] ✗ 输出不符合预期: {first_line[:80]}")
                    total_fail += 1
            else:
                print(f"  [{fname}] ✗ 运行失败 (exit={retcode})")
                if output:
                    print(f"    输出: {output[:200]}")
                total_fail += 1

            # Clean up binary
            binary = src.with_suffix("")
            if binary.exists():
                binary.unlink()

        print()

    print(f"{'='*60}")
    print(f"总计: {total_ok} 通过, {total_fail} 失败, {total_skip} 跳过/无法判断")
    print(f"{'='*60}")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
