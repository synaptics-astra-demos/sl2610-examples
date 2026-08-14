# app_utils

Shared helper code for the sl2610-examples demos including the 
[`torq_examples`](torq_examples/) git submodule.

## Namespace note: this package is deliberately *not* named `utils`

`torq_examples` is a standalone project (developed and versioned in its own
repo) that has its own internal `utils` package (`torq_examples/utils/`). Its
code imports that with bare names, e.g. `from utils.download import ...`,
because it's designed to be run with its own directory as a `sys.path` root.

Naming this package to `app_utils` fixes the top-level naming collision for
good: `app_utils` always means this package, and bare `utils` (used only
inside `torq_examples`'s own files, via the `sys.path` entry added in
[`__init__.py`](__init__.py)) always means `torq_examples`'s own bundled copy.
That fix holds as long as:

- **Don't rename this package back to `utils`**, and don't add a bare
  `utils/` package anywhere else on `sys.path` in this project.

That fix does *not* cover a second, separate risk: a name used somewhere
inside `app_utils/` happening to match a name `torq_examples` already uses
internally. Rules for that, going forward:

- **Allowed:** add any new file or module to `app_utils/`, as long as its name
  doesn't already exist inside `torq_examples/`.
- **Not allowed:** giving something in `app_utils/` the same name as something
  already inside `torq_examples/` - that would let this package silently
  shadow `torq_examples`'s own module for anything using a bare import.
- **Out of scope** if `torq_examples` later adds a file that collides with 
 `app_utils` or with anything else on `sys.path`, we can't prevent that from this side.

## The one remaining gotcha: dual module identity

Our own code reaches into `torq_examples` via explicit dotted imports, e.g.:

```python
from app_utils.torq_examples.utils.download import DownloadError
```

`torq_examples`'s own internal code reaches the *same file* via its bare
import instead, e.g. `moonshine/setup_demo.py` does `from utils.download
import DownloadError`. Two different import paths to one file means Python
loads it twice, under two different module names - so there end up being
**two separate `DownloadError` classes**, not one (same for `ModelStatus`,
or anything else defined in a module reached both ways).

This is a correctness risk, not a performance one: `except DownloadError:` or
`isinstance(e, DownloadError)` written against one import path will silently
fail to match an exception raised via the other path - the error isn't
caught, it just propagates as if no handler existed.

This is not currently a live bug: nothing in this codebase catches
`torq_examples` exceptions by specific type across that boundary - callers use
a generic `except Exception:` around calls into `torq_examples` setup
functions (see `app_utils/yolo_od/download.py`). If you ever need to catch a
`torq_examples`-raised exception by its specific class, catch it via the same
import path the raising code used (bare `utils.download.DownloadError`, not
the dotted one), or ask whether generic `Exception` handling is good enough
first.
