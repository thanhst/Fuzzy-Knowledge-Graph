# FKG Python Module

This folder owns Python-side adapters around FKG.

- `fkg_runtime/module_loader.py`: shared resolver/importer for the native
  `fisa_module` extension.
- New runners should import from `modules.fkg_python` and should not hard-code
  `Source`, `GPU/Source`, or Windows DLL paths.

Legacy pure-Python experiments are still under `Source_code/module/FKG`. Keep
that tree stable for old scenarios, and place new reusable Python runtime code
here.
