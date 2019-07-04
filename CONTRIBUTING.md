# Coding style
Traffic4Cast code conforms to [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

## Code formatter.
For code formating [yapf](https://github.com/google/yapf) is used to enforce
the chosen coding style.
Please run **yapf** on your files with `--style google` flag.
```bash
yapf --style google --no-local-style src/dataset.py
```
For **vim**/**neovim** users, both [yapf](https://github.com/google/yapf/tree/master/plugins)
and [Google Python Style Guide](http://google.github.io/styleguide/google_python_style.vim)
have plugins to help with code formating.

## Type hint checker - experimental.
Type hint should be checked with [pytype](https://github.com/google/pytype).
At the moment it is troublesome to install it in the anaconda environment.
([pytype issue](https://github.com/conda/conda/issues/8648))
Because of that import checking of module libraries might not work. Ignore those
type of errors for now.
```bash
pytype-single src/dataset.py
```

## Pylint
[Pylint](https://www.pylint.org/) usage is encouragend.

# Pull requests (PRs)
All contributions must have an associated PR with at least one reviewer. Please
not that our gitlab server does not support adding reviewers. The work around
is to indicate the reviewer(s) with @. In order to merge all discussions must
be resolved and the reviers(s) must mark with LGTM (looks good to me) for
acceptance.

**Note: Any member can review and it is encouraged to do so.**

