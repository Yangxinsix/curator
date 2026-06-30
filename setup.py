from __future__ import annotations

from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "curator.native._neighbors",
            sources=["curator/native/neighbors.cpp"],
            language="c++",
            extra_compile_args=["-std=c++17", "-O3"],
        )
    ]
)
