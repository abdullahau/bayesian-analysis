# Quarto Help

Source: 

- [Quarto: JupyterLab](https://quarto.org/docs/tools/jupyter-lab.html)
- [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html)
- [Jupytext: Quarto](https://jupytext.readthedocs.io/en/latest/formats-markdown.html#quarto)

## Converting Notebook

```
quarto convert basics-jupyter.ipynb # converts to qmd
quarto convert basics-jupyter.qmd   # converts to ipynb
```

## Jupytext

### Pair a Notebook (`.ipynb`) With `.qmd`

```
jupytext --set-formats ipynb,qmd <filename>.ipynb
```

### Synchronize the paired files

Sync changes made in `.ipynb` to `.qmd`

```
jupytext --sync <filename>.qmd
```

Sync changes made in `.qmd` to `.ipynb`

```
jupytext --sync <filename>.ipynb
```

### Others

convert a notebook in one format to another with `jupytext --to ipynb notebook.py` (use `-o` if you want a specific output file)

pipe a notebook to a linter with e.g. `jupytext --pipe black notebook.ipynb`