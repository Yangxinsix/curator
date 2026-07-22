from __future__ import annotations

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Apply compiler-specific C++17 flags to native extensions."""

    def build_extensions(self) -> None:
        compiler = self.compiler.compiler_type
        if compiler == "msvc":
            compile_args = ["/std:c++17", "/O2", "/EHsc"]
            link_args = []
        elif compiler in {"unix", "mingw32", "cygwin"}:
            compile_args = ["-std=c++17", "-O3"]
            link_args = []
            if sys.platform.startswith("linux"):
                compile_args.append("-pthread")
                link_args.append("-pthread")
        else:
            raise RuntimeError(f"Unsupported C++ compiler: {compiler}")

        for extension in self.extensions:
            extension.extra_compile_args = [
                *(extension.extra_compile_args or []),
                *compile_args,
            ]
            extension.extra_link_args = [
                *(extension.extra_link_args or []),
                *link_args,
            ]
        super().build_extensions()


setup(
    ext_modules=[
        Extension(
            "curator.native._neighbors",
            sources=["curator/native/neighbors.cpp"],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
