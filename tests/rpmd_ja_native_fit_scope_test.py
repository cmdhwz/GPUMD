from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIT = (ROOT / "src/measure/rpmd_ja_native_fit.cu").read_text(encoding="utf-8")


def brace_pairs(text):
    pairs = {}
    stack = []
    unterminated = []
    state = "code"
    raw_end = ""
    index = 0
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 2
                continue
            index += 1
            continue
        if state == "raw_string":
            if text.startswith(raw_end, index):
                state = "code"
                index += len(raw_end)
                continue
            index += 1
            continue
        if state in {"string", "character"}:
            if char == "\\":
                index += 2
                continue
            if (state == "string" and char == '"') or (state == "character" and char == "'"):
                state = "code"
            index += 1
            continue
        if char == "/" and following == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and following == "*":
            state = "block_comment"
            index += 2
            continue
        if char == "R" and following == '"':
            delimiter_end = text.find("(", index + 2)
            if delimiter_end >= 0:
                raw_end = ")" + text[index + 2:delimiter_end] + '"'
                state = "raw_string"
                index = delimiter_end + 1
                continue
        if char == '"':
            state = "string"
        elif char == "'":
            state = "character"
        elif char == "{":
            stack.append(index)
        elif char == "}":
            if not stack:
                unterminated.append(index)
            else:
                pairs[stack.pop()] = index
        index += 1
    return pairs, stack, unterminated


def test_feasibility_and_native_function_scopes_remain_open_through_their_consumers():
    pairs, unclosed, underflows = brace_pairs(FIT)
    assert not underflows, "native fit source closes a scope before it was opened"
    assert not unclosed, "native fit source leaves a scope open"

    function_start = FIT.index("static bool fit_rpmd_ja_native_reference_impl(")
    function_open = FIT.index("{", function_start)
    function_close = pairs[function_open]
    next_function = FIT.index("static bool fit_rpmd_ja_native_reference_checked(", function_start)
    assert function_close < next_function, "native reference implementation closes after the next function begins"
    assert "return !saved_recovery_witness_paths.empty();" in FIT[function_open:function_close]

    feasibility_marker = "if(covariance_method&&options.covariance_feasibility){"
    feasibility_if = FIT.index(feasibility_marker, function_open, function_close)
    feasibility_open = feasibility_if + len(feasibility_marker) - 1
    feasibility_close = pairs[feasibility_open]
    non_feasibility = FIT.index("if(covariance_method&&!options.covariance_feasibility){", feasibility_open)
    blind_commit = FIT.index("if(blind_response_pass_completed)", feasibility_open)
    blind_restore = FIT.index("if(restored_blind_response_present)", blind_commit)
    assert blind_commit < blind_restore < feasibility_close < non_feasibility
    assert "bool blind_response_pass_completed=false" in FIT[feasibility_open:feasibility_close]
    assert "completed_blind_response_sha256=write_blind_response_checkpoint" in FIT[feasibility_open:feasibility_close]

    outer_persist = FIT.index("auto persist_publication_controller_journal=[&](){", function_open, feasibility_open)
    assert function_open < outer_persist < feasibility_if
    publication_calls = [position for position in range(function_open, function_close)
                         if FIT.startswith("persist_publication_controller_journal();", position)]
    assert len(publication_calls) >= 5 and all(position > feasibility_close for position in publication_calls)

    publisher_start = FIT.index("void publish_verified_prepared_artifacts(")
    publisher_end = FIT.index("std::string normalized_sha256(", publisher_start)
    publisher = FIT[publisher_start:publisher_end]
    assert "const bool final_exists=[&](){rpmd_ja_file_io::ifstream final(final_path,std::ios::binary);return final.good();}();" in publisher
    assert "const bool stage_exists=[&](){rpmd_ja_file_io::ifstream stage(staged_path,std::ios::binary);return stage.good();}();" in publisher
    assert "input.close();return fit_input_content_identity(staged.first,true).sha256==staged.second;" in publisher
    assert "const bool committed_fit_exists=[&](){rpmd_ja_file_io::ifstream committed_fit(fit_path,std::ios::binary);" in FIT
    assert "duplicate_fit.close();return fit_input_content_identity(fit_tmp,true).sha256==" in FIT

    integration = (ROOT / "tests/rpmd_ja_fit_prepare_integration_cuda_test.cu").read_text(encoding="utf-8")
    assert "prepare_rpmd_ja_qnep_reference(raw,transaction_stage" in integration
    assert "publish_verified_prepared_artifacts(transaction_output,transaction_stage" in integration
    assert "two_rename_interruption" in integration


if __name__ == "__main__":
    test_feasibility_and_native_function_scopes_remain_open_through_their_consumers()
    print("native fit lexical scope and consumer visibility checks passed")
