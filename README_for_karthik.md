# README for Karthik: what was done, step by step, in plain words

## Read this first (honest part)

Claude wrote this code. A GitHub repo you cannot explain is worth nothing in an interview. The value of Week 1 is that **you** can explain every file. So: read each section below, run each command yourself, then close the file and retype one piece from memory (start with `vector.py`, then `engine.py`). That is the actual work.

---

## Part A: What I did (in order)

### Step 1: Read your PDF
The PDF is a 7-day plan. Each day builds one piece of a small math library called **tiny-vec**. I followed it exactly, except where it was wrong or incomplete (listed in Part D).

### Step 2: Built the library (folder `tiny_vec/`)

| File | Layman explanation | Why it matters for AI jobs |
|---|---|---|
| `vector.py` | A list of numbers that can add, subtract, and compute "dot product" (multiply matching numbers, add them up) and length. | Every AI model is dot products. Embeddings, attention, similarity search: all dot products. |
| `matrix.py` | A grid of numbers. Can flip (transpose) and multiply two grids. Two ways to multiply: the normal slow way, and **Strassen** (a cleverer way with fewer multiplications). | Neural networks are matrix multiplications stacked together. |
| `engine.py` | The "autograd" engine. Each number remembers how it was made. After a calculation, you press `backward()` and it works out how much each input contributed to the result (the **gradient**). | This is exactly how PyTorch trains models. If you understand this file, you understand backpropagation. |
| `nn.py` | A tiny neural network (Neuron, Layer, MLP) built only from the engine above. | Shows you can build a network from nothing, not just call a library. |
| `linalg.py` | **SVD** and **PCA** done by hand using "power iteration" (repeatedly multiply a vector by a matrix until it settles on the strongest direction). PCA squashes many-column data into 2 columns while keeping the most information. | PCA/SVD show up in embeddings, compression, LoRA, recommenders. |
| `__init__.py` | Makes `from tiny_vec import Value` work. | Packaging. |

### Step 3: Wrote tests (folder `tests/`)
A test = a small program that checks my math against NumPy (a trusted library). If my answer matches NumPy's, I trust mine.
- `test_core.py`: vector and matrix checks, including one **property test** (the `hypothesis` library throws hundreds of random inputs at the code to catch edge cases).
- `test_math.py`: checks that the autograd gradient matches a numerical estimate, and that my SVD matches NumPy's.

Result: **11 tests pass, 80% of the code is covered by tests.**

### Step 4: Ran experiments (folders `examples/` and `bench/`)
- `train_xor.py`: teaches the tiny network the XOR puzzle (0,0→0, 0,1→1, 1,0→1, 1,1→0). A single neuron cannot do this; you need a hidden layer. Loss went to about 0, so it learned it.
- `pca_iris.py`: squashes the famous iris flower dataset to 2 dimensions with my own PCA and plots it next to scikit-learn's. They look the same. Plot is `docs/iris_pca.png`.
- `bench/matmul.py`: times slow multiply vs Strassen.

### Step 5: The benchmark results (this is the interesting part)
- NumPy was about **229x faster** than my matrix multiply (128x128). Reason: NumPy runs optimized C code; plain Python has overhead on every single operation.
- Strassen is supposed to beat the normal method on big matrices. In my run it **lost at 128 and 256 and only won at 512**. The PDF claimed it wins around 256. On this machine it did not, so I wrote the real numbers in the README instead of copying the PDF. Being honest about a result that contradicts the textbook is what makes this project credible.

### Step 6: Packaged it and set up automation
- `pyproject.toml`: makes it installable (`pip install -e .`).
- `.github/workflows/ci.yml`: every time you push, GitHub automatically runs the tests on Python 3.10, 3.11, 3.12. A green tick on GitHub proves it works.
- `.github/workflows/pages.yml`: automatically publishes `docs/index.html` as a free website (GitHub Pages).
- `.gitignore`: keeps junk files out of git.

### Step 7: Wrote the docs
- `README.md`: the public project page with the benchmark table.
- `docs/blog.md`: the ~300-word blog post the PDF asked for, using the real numbers.

### Step 8: Committed and pushed
6 commits pushed to branch `claude/adoring-fermi-je5toz` on your repo.

---

## Part B: What I need from you (3 things)

1. **Merge to `main`** (GitHub → Pull requests → New → base `main`, compare `claude/adoring-fermi-je5toz` → Create → Merge). The CI and website workflows only run on `main`. Tell me if you want me to open the pull request.
2. **Turn on the free website:** GitHub repo → Settings → Pages → Source → choose **GitHub Actions**. After the next push to `main`, your site is at `https://kashyapphassan2000-netizen.github.io/AI-DOUNDATIONS-WEEK-01/`.
3. **Optional, TestPyPI publish:** create an account at test.pypi.org, make an API token, then run `pip install build twine && python -m build && twine upload --repository testpypi dist/*`. I cannot do this: it needs your private token, and you should never paste a token into chat.

Why I need these: I am only allowed to push to the one feature branch, and repo settings plus account tokens belong to you.

---

## Part C: How to run it yourself

```bash
git clone https://github.com/kashyapphassan2000-netizen/AI-DOUNDATIONS-WEEK-01.git
cd AI-DOUNDATIONS-WEEK-01
git checkout claude/adoring-fermi-je5toz
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                          # expect: 11 passed
python examples/train_xor.py       # expect: PASS: loss < 0.01
python examples/pca_iris.py        # makes docs/iris_pca.png
python bench/matmul.py --sizes 128,256,512 --leaf 32   # your own timings (512 takes ~20s)
```

Your timings will differ from mine. Put YOUR numbers in the README if they differ.

---

## Part D: Things I changed vs. the PDF
- The PDF's `pyproject.toml` needs a `README.md` to exist or install fails. I created it.
- The PDF's benchmark numbers were from someone else's laptop. I used mine.
- `authors` is your GitHub handle because the PDF said "Karthik" and I would not guess your full name. Edit `pyproject.toml` if you want.
- I did not do 7 daily commits over 7 days (the PDF's idea). There are 6 commits made in one sitting. Do not claim in an interview that you built this over a week unless you actually re-do it that way.

---

## Part E: What you can say in an interview (only if you really understand it)
- "I implemented dot product, matmul and transpose from scratch and checked them against NumPy."
- "I measured a ~229x gap to NumPy and can explain why: C/BLAS vs interpreter overhead."
- "Strassen did 7 multiplies instead of 8 but lost to the plain loop below n=512 in pure Python, because extra additions and list copying cost more than the saved multiply."
- "I wrote reverse-mode autodiff: build a graph, sort it, apply the chain rule backwards, accumulate gradients with `+=`."
- "I did SVD via power iteration plus deflation, and PCA on top; it matched scikit-learn."

If you cannot explain one of these lines, do not say it. Go back to the file and re-read it.

## Part F: Honest reality about jobs
This one project does not get you hired. It is a learning step. Employers pay for people who ship working AI products (LLM apps, RAG, evals, data pipelines). Do the 12 weeks, but from about Week 4 onward put real, deployed projects with real users or real data on your profile. Week 2 onward I can build as you ask.
