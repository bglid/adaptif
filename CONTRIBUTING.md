# How to contribute

### READ BELOW

---

#### If a PR is submitted with no issue attached and a green light on working on a PR, it will be generally closed. Undisclosed AI PRs will be **closed** automatically

#### **Overall, submit an issue instead.**

---

## Dependencies

We use `uv` to manage Python dependencies and the dev env.
If you dont have `uv`, please install that first then install the project and dev dependencies:

```bash
uv sync
```

Install [`pre-commit`](https://pre-commit.com/) hooks:

```bash
uv run pre-commit install
```

`uv` automatically manages the project virtual environment in `.venv`. Manually activating it is unnecessary. Run project tools with `uv run`.

## Python bindings

To build the Python bindings for the Rust code, use `maturin`:

```bash
uv run maturin develop
```

The build artifacts will be placed in `adaptif/`

### Adding new bindings

New bindings should be imported in `adaptif/__init__.py`.
In order for `ty` to resolve imports correctly, you also have too add any new functions/classes to the stub file (`adaptif/adaptif.pyi`).

## Codestyle

Format & lint the code with:

```bash
uv run ruff check --fix .
uv run ruff format .
```

### Checks

To run the test suite:

```bash
uv run pytest
```

To run the ty type checking:

```bash
uv run ty check
```

To run the security checks:

```bash
uv audit
```

To run the pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

### File tracking

We maintain `.gitignore` as a whitelist by ignoring all files by default and implicitly including only the files we actually need by prefixing them with `!`, e.g.:

```bash
# directory
!src/

# all subdirectories
!src/**/

# only Rust files
!src/**/*.rs
```

If you add new files to `.gitignore`, try to keep your additions reasonably concise by using wildcard like in the example above.
Likewise, if you delete any files, make sure to remove any lines that are no longer necessary.

### Before submitting

**Again, READ the section at the [top](#if-a-pr-is-submitted-with-no-issue-attached-and-a-green-light-on-working-on-a-pr-it-will-be-generally-closed-ai-prs-will-be-closed-automatically)**

Before submitting your code please do the following steps:

1. Add tests for new changes
   - _Update documentation for significant changes._
2. Update `.gitignore` to whitelist any files you created and remove any files you deleted.
3. Format your changes with `ruff`
4. Run `uv run pytest`
5. Run `uv run ty check`
6. Run `uv run pre-commit run --all-files`
7. Commit any changes to `uv.lock` if you modified project dependencies

## Other help

You can contribute by spreading a word about this library.
You can also share your best practices with us.

---

**In particular**, if you use this in any DSP research, please let us know!!

1. Because we love the topic and would love to check out and share your research
2. It gives us an opportunity to see how this library is being used and how it can be improved.

---
