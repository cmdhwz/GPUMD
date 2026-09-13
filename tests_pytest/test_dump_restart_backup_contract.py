from pathlib import Path


ROOT = Path(__file__).parents[1]


def _function_body(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for position in range(brace, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError(f"unclosed function: {signature}")


def test_dump_restart_backup_directory_preserves_latest_root_restart():
    source = (ROOT / "src/measure/dump_restart.cu").read_text(encoding="utf-8")
    header = (ROOT / "src/measure/dump_restart.cuh").read_text(encoding="utf-8")
    parse = _function_body(source, "void Dump_Restart::parse(")
    pre_run = _function_body(source, "void Dump_Restart::pre_run(")
    end_of_step = _function_body(source, "void Dump_Restart::end_of_step(")

    assert '"utilities/gpu_macro.cuh"' not in source
    assert "bool backup_ = false;" in header
    assert "std::string backup_directory_;" in header
    assert "num_param != 2 && num_param != 3" in parse
    assert "is_valid_backup_directory_name" in parse
    assert "backup_directory_ = param[2]" in parse
    assert "make_restart_backup_directory(backup_directory_)" in pre_run
    assert 'filename = "restart.xyz"' in end_of_step
    assert 'backup_filename << backup_directory_ << "/restart_step_"' in end_of_step
    assert "std::setw(10)" in end_of_step
    assert "std::setfill('0')" in end_of_step
    assert 'copy_restart_file(filename, "restart.xyz")' in end_of_step


def test_dump_restart_backup_documentation_matches_the_file_naming_contract():
    docs = (ROOT / "doc/gpumd/input_parameters/dump_restart.rst").read_text(encoding="utf-8")
    assert "dump_restart <interval> [backup_directory]" in docs
    assert "restart_step_0000100000.xyz" in docs
    assert "restart.xyz" in docs
