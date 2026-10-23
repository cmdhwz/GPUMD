import shutil


def find_cpp_compiler():
    for name in ("c++", "g++", "clang++", "cl"):
        compiler = shutil.which(name)
        if compiler:
            return compiler, name
    return None


def compile_command(compiler, standard, source, executable):
    path, name = compiler
    if name == "cl":
        return [path, "/nologo", f"/std:c++{max(14, standard)}", "/EHsc", "/utf-8", "/W4",
                f"/Fe:{executable}", str(source)]
    return [path, f"-std=c++{standard}", "-Wall", "-Wextra", "-pedantic",
            str(source), "-o", str(executable)]
