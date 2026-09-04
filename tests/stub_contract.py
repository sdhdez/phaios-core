# SPDX-License-Identifier: GPL-3.0-or-later
"""Stdlib-only checker that the `.pyi` stubs in `python/phaios_core/`
mirror the runtime module exactly: public names, function/method/`__new__`
signatures (including the real value behind every non-literal `...`
default), class shape (enum variants, get/set fields, methods, `__new__`
presence, unhashability), and docstrings (verbatim, module through field).
Also checks that the installed package actually ships the stub files and
that the installed `__init__.pyi` matches the repo copy byte for byte.

Shared by `tests/ffi.py` (the top-level `phaios_core` module) and
`tests/ffi_gpu.py` (the optional `phaios_core.gpu` submodule).

**Deliberately not checked** (see the type-stubs plan): setter presence
on a field that is get-only at runtime — that needs an instance of a
class this module cannot always construct — and the *types* written in
the stub, which `mypy --strict` covers in a later step of the same plan.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import ModuleType

import phaios_core as ph

_EMPTY = inspect.Parameter.empty


# ── AST helpers ──────────────────────────────────────────────────────────


def parse_stub(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), str(path))


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _stub_arg_names(args: ast.arguments) -> list[str]:
    return [a.arg for a in args.args]


def _stub_has_default(args: ast.arguments) -> list[bool]:
    n_no_default = len(args.args) - len(args.defaults)
    return [False] * n_no_default + [True] * len(args.defaults)


def _is_property(fn: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "property" for d in fn.decorator_list)


def _class_scalar_attrs(cls_node: ast.ClassDef) -> dict[str, str | None]:
    """Annotated scalar attributes (`name: Type` followed by a docstring
    string literal), as {name: docstring_or_None}."""
    out: dict[str, str | None] = {}
    body = cls_node.body
    for i, item in enumerate(body):
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            if name == "__hash__":
                continue
            doc = None
            if (
                i + 1 < len(body)
                and isinstance(body[i + 1], ast.Expr)
                and isinstance(body[i + 1].value, ast.Constant)
                and isinstance(body[i + 1].value.value, str)
            ):
                doc = body[i + 1].value.value
            out[name] = doc
    return out


def _class_properties(cls_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    """Property getters (the `@property`-decorated methods), as
    {name: getter_FunctionDef}."""
    return {
        item.name: item
        for item in cls_node.body
        if isinstance(item, ast.FunctionDef) and _is_property(item)
    }


def _class_methods(cls_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    """Plain (non-property, non-setter, non-dunder, non-`__new__`) methods."""
    property_names = set(_class_properties(cls_node))
    return {
        item.name: item
        for item in cls_node.body
        if isinstance(item, ast.FunctionDef)
        and not _is_property(item)
        and item.name != "__new__"
        and not _is_dunder(item.name)
        and item.name not in property_names  # excludes the matching `@x.setter`
    }


def _class_new(cls_node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__new__":
            return item
    return None


def _class_has_hash_none(cls_node: ast.ClassDef) -> bool:
    return any(
        isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and item.target.id == "__hash__"
        for item in cls_node.body
    )


def _is_runtime_enum(cls: type) -> bool:
    """True if `cls` has class attributes that are instances of itself
    (PyO3's rendering of a `#[pyclass(eq, eq_int)] enum`)."""
    return any(
        not name.startswith("_") and isinstance(val, cls) for name, val in vars(cls).items()
    )


def _runtime_getset_names(cls: type) -> set[str]:
    return {
        name
        for name, val in vars(cls).items()
        if not name.startswith("_") and type(val).__name__ == "getset_descriptor"
    }


def _runtime_method_names(cls: type) -> set[str]:
    return {
        name
        for name, val in vars(cls).items()
        if not name.startswith("_")
        and type(val).__name__ != "getset_descriptor"
        and (inspect.isroutine(val) or callable(val))
        and not isinstance(val, cls)  # exclude enum variants
    }


def _runtime_enum_variant_names(cls: type) -> set[str]:
    return {name for name, val in vars(cls).items() if not name.startswith("_") and isinstance(val, cls)}


# ── Group 1: names ───────────────────────────────────────────────────────


def check_names(tree: ast.Module, module: ModuleType, *, top_level: bool) -> list[str]:
    errors: list[str] = []
    runtime_all = list(getattr(module, "__all__", []))
    runtime_public = [n for n in runtime_all if not _is_dunder(n)]

    if "gpu" in runtime_public:
        runtime_public.remove("gpu")

    stub_top_names = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }

    missing = sorted(set(runtime_public) - stub_top_names)
    extra = sorted(stub_top_names - set(runtime_public))
    if missing:
        errors.append(f"names: runtime __all__ has {missing} missing from the stub")
    if extra:
        errors.append(f"names: stub declares {extra}, not present in runtime __all__")

    if top_level:
        has_version = any(
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "__version__"
            and isinstance(n.annotation, ast.Name)
            and n.annotation.id == "str"
            for n in tree.body
        )
        if not has_version:
            errors.append("names: stub is missing `__version__: str`")

        has_gpu_import = any(
            isinstance(n, ast.ImportFrom)
            and n.module is None
            and any(a.name == "gpu" and a.asname == "gpu" for a in n.names)
            for n in tree.body
        )
        if not has_gpu_import:
            errors.append("names: stub is missing `from . import gpu as gpu`")

    return errors


# ── Group 2: signatures ──────────────────────────────────────────────────


def _is_valid_ellipsis_default(node: ast.expr) -> bool:
    if isinstance(node, ast.List):
        return True
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        enum_cls = getattr(ph, node.value.id, None)
        return isinstance(enum_cls, type) and _is_runtime_enum(enum_cls)
    return False


def _check_defaults(
    label: str,
    stub_names: list[str],
    stub_defaults: list[ast.expr],
    runtime_params: list[inspect.Parameter],
    *,
    cross_check_cls: type | None,
    errors: list[str],
) -> None:
    n_no_default = len(stub_names) - len(stub_defaults)
    for i, default_node in enumerate(stub_defaults):
        pname = stub_names[n_no_default + i]
        rparam = next((p for p in runtime_params if p.name == pname), None)
        if rparam is None:
            continue
        rdefault = rparam.default
        if rdefault is Ellipsis:
            if not _is_valid_ellipsis_default(default_node):
                errors.append(
                    f"{label}: default for `{pname}` renders `...` at runtime; the stub "
                    f"default must be a stub-enum attribute or a list literal, got "
                    f"{ast.dump(default_node)}"
                )
                continue
            if cross_check_cls is not None:
                try:
                    stub_val = eval(  # noqa: S307 - trusted repo source, not user input
                        compile(ast.Expression(body=default_node), "<stub-default>", "eval"),
                        {},
                        dict(vars(ph)),
                    )
                    inst = cross_check_cls()
                    actual = getattr(inst, pname)
                    if stub_val != actual:
                        errors.append(
                            f"{label}: default for `{pname}` evaluates to {stub_val!r}, but "
                            f"{cross_check_cls.__name__}().{pname} is {actual!r}"
                        )
                except Exception as exc:  # pragma: no cover - reported, not swallowed
                    errors.append(f"{label}: could not cross-check default for `{pname}`: {exc}")
        else:
            try:
                stub_val = ast.literal_eval(default_node)
            except ValueError:
                errors.append(
                    f"{label}: default for `{pname}` is not a literal in the stub, but "
                    f"runtime default is the literal {rdefault!r}"
                )
                continue
            if stub_val != rdefault:
                errors.append(
                    f"{label}: default for `{pname}` is {stub_val!r} in the stub, "
                    f"{rdefault!r} at runtime"
                )


def _check_function_like(
    label: str,
    stub_fn: ast.FunctionDef,
    runtime_callable,
    *,
    drop_leading: str | None,
    cross_check_cls: type | None,
    errors: list[str],
) -> None:
    if runtime_callable is None:
        errors.append(f"{label}: not found at runtime")
        return
    try:
        sig = inspect.signature(runtime_callable)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: runtime signature unavailable: {exc}")
        return
    runtime_params = list(sig.parameters.values())
    runtime_names = [p.name for p in runtime_params]
    runtime_has_default = [p.default is not _EMPTY for p in runtime_params]

    stub_names_all = _stub_arg_names(stub_fn.args)
    stub_has_default_all = _stub_has_default(stub_fn.args)
    if drop_leading is not None:
        if not stub_names_all or stub_names_all[0] != drop_leading:
            errors.append(f"{label}: stub is missing leading `{drop_leading}` parameter")
            return
        stub_names = stub_names_all[1:]
        stub_has_default = stub_has_default_all[1:]
    else:
        stub_names = stub_names_all
        stub_has_default = stub_has_default_all

    if stub_names != runtime_names:
        errors.append(f"{label}: parameter names {stub_names} != runtime {runtime_names}")
        return
    if stub_has_default != runtime_has_default:
        errors.append(
            f"{label}: default-presence {stub_has_default} != runtime {runtime_has_default}"
        )
        return

    _check_defaults(
        label,
        stub_names_all,
        stub_fn.args.defaults,
        runtime_params,
        cross_check_cls=cross_check_cls,
        errors=errors,
    )


def check_signatures(tree: ast.Module, module: ModuleType) -> list[str]:
    errors: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            _check_function_like(
                node.name,
                node,
                getattr(module, node.name, None),
                drop_leading=None,
                cross_check_cls=None,
                errors=errors,
            )
        elif isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name, None)
            if cls is None:
                continue
            if _is_runtime_enum(cls):
                continue  # enums declare no __new__ and no methods to check here

            new_node = _class_new(node)
            runtime_has_new = getattr(cls, "__text_signature__", None) is not None
            if (new_node is not None) != runtime_has_new:
                errors.append(
                    f"{node.name}: stub __new__ present={new_node is not None}, "
                    f"runtime __text_signature__ is not None={runtime_has_new}"
                )
            elif new_node is not None:
                all_defaulted = all(
                    p.default is not _EMPTY for p in inspect.signature(cls).parameters.values()
                )
                _check_function_like(
                    f"{node.name}.__new__",
                    new_node,
                    cls,
                    drop_leading="cls",
                    cross_check_cls=cls if all_defaulted else None,
                    errors=errors,
                )

            for mname, mnode in _class_methods(node).items():
                runtime_method = getattr(cls, mname, None)
                _check_function_like(
                    f"{node.name}.{mname}",
                    mnode,
                    runtime_method,
                    drop_leading=None,  # runtime unbound-method signatures already include `self`
                    cross_check_cls=None,
                    errors=errors,
                )
    return errors


# ── Group 3: classes ─────────────────────────────────────────────────────


def check_classes(tree: ast.Module, module: ModuleType) -> list[str]:
    errors: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        cls = getattr(module, node.name, None)
        if cls is None:
            errors.append(f"classes: {node.name} not found at runtime")
            continue

        if _is_runtime_enum(cls):
            stub_variants = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id != "__hash__"
            }
            runtime_variants = _runtime_enum_variant_names(cls)
            if stub_variants != runtime_variants:
                errors.append(
                    f"{node.name}: enum variants {sorted(stub_variants)} != "
                    f"runtime {sorted(runtime_variants)}"
                )
        else:
            stub_attr_names = set(_class_scalar_attrs(node)) | set(_class_properties(node))
            runtime_getset = _runtime_getset_names(cls)
            missing = sorted(runtime_getset - stub_attr_names)
            extra = sorted(stub_attr_names - runtime_getset)
            if missing:
                errors.append(f"{node.name}: runtime fields {missing} missing from the stub")
            if extra:
                errors.append(f"{node.name}: stub declares fields {extra} absent at runtime")

            stub_method_names = set(_class_methods(node))
            runtime_methods = _runtime_method_names(cls)
            missing_m = sorted(runtime_methods - stub_method_names)
            extra_m = sorted(stub_method_names - runtime_methods)
            if missing_m:
                errors.append(f"{node.name}: runtime methods {missing_m} missing from the stub")
            if extra_m:
                errors.append(f"{node.name}: stub declares methods {extra_m} absent at runtime")

        new_node = _class_new(node)
        runtime_has_new = getattr(cls, "__text_signature__", None) is not None
        if (new_node is not None) != runtime_has_new:
            errors.append(
                f"{node.name}: stub __new__ present={new_node is not None} but runtime "
                f"__text_signature__ is not None={runtime_has_new}"
            )

        stub_hash_none = _class_has_hash_none(node)
        runtime_hash_none = cls.__hash__ is None
        if stub_hash_none != runtime_hash_none:
            errors.append(
                f"{node.name}: stub `__hash__: ClassVar[None]` present={stub_hash_none}, "
                f"runtime __hash__ is None={runtime_hash_none}"
            )
    return errors


# ── Group 4: docstrings ──────────────────────────────────────────────────


def check_docstrings(tree: ast.Module, module: ModuleType) -> list[str]:
    errors: list[str] = []

    mod_doc = ast.get_docstring(tree, clean=False)
    if mod_doc != module.__doc__:
        errors.append("docstrings: module docstring does not match runtime __doc__")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = getattr(module, node.name, None)
            if fn is None:
                continue
            stub_doc = ast.get_docstring(node, clean=False)
            if stub_doc != fn.__doc__:
                errors.append(f"docstrings: function `{node.name}` does not match runtime __doc__")
        elif isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name, None)
            if cls is None:
                continue
            stub_doc = ast.get_docstring(node, clean=False)
            if stub_doc != cls.__doc__:
                errors.append(f"docstrings: class `{node.name}` does not match runtime __doc__")

            if _is_runtime_enum(cls):
                continue

            for fname, fdoc in _class_scalar_attrs(node).items():
                runtime_attr = getattr(cls, fname, None)
                if runtime_attr is None:
                    continue
                if fdoc != runtime_attr.__doc__:
                    errors.append(
                        f"docstrings: `{node.name}.{fname}` attribute does not match "
                        f"runtime getset_descriptor.__doc__"
                    )

            for pname, pnode in _class_properties(node).items():
                runtime_attr = getattr(cls, pname, None)
                if runtime_attr is None:
                    continue
                pdoc = ast.get_docstring(pnode, clean=False)
                if pdoc != runtime_attr.__doc__:
                    errors.append(
                        f"docstrings: `{node.name}.{pname}` property getter does not match "
                        f"runtime getset_descriptor.__doc__"
                    )

            for mname, mnode in _class_methods(node).items():
                runtime_attr = getattr(cls, mname, None)
                if runtime_attr is None:
                    continue
                mdoc = ast.get_docstring(mnode, clean=False)
                if mdoc != runtime_attr.__doc__:
                    errors.append(
                        f"docstrings: `{node.name}.{mname}` method does not match runtime __doc__"
                    )
    return errors


# ── Group 5: packaging ───────────────────────────────────────────────────


def check_packaging(module: ModuleType, repo_root: Path) -> list[str]:
    """`module` must be the top-level `phaios_core` module (its
    `__file__` locates the installed package directory)."""
    errors: list[str] = []
    installed_dir = Path(module.__file__).parent

    for fname in ("__init__.pyi", "gpu.pyi", "py.typed"):
        if not (installed_dir / fname).is_file():
            errors.append(f"packaging: installed package is missing {fname}")

    installed_init_pyi = installed_dir / "__init__.pyi"
    repo_init_pyi = repo_root / "python" / "phaios_core" / "__init__.pyi"
    if installed_init_pyi.is_file() and repo_init_pyi.is_file():
        if installed_init_pyi.read_bytes() != repo_init_pyi.read_bytes():
            errors.append(
                "packaging: installed __init__.pyi differs from the repo copy "
                "(stale `maturin develop`?)"
            )
    return errors


# ── Entry point ───────────────────────────────────────────────────────────


def check_stub(stub_path: Path, module: ModuleType, *, top_level: bool) -> list[str]:
    """Run the name/signature/class/docstring checks (groups 1-4) for one
    stub file against its runtime module. Returns a list of discrepancy
    strings; empty means the stub matches."""
    tree = parse_stub(stub_path)
    errors: list[str] = []
    errors += check_names(tree, module, top_level=top_level)
    errors += check_signatures(tree, module)
    errors += check_classes(tree, module)
    errors += check_docstrings(tree, module)
    return errors
