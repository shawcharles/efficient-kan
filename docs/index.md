---
title: efficient-kan Documentation
---

<section class="hero">
  <div>
    <p class="eyebrow">Pure PyTorch KAN layers</p>
    <h1>Documentation for reproducible KAN research workflows.</h1>
    <p class="hero-lede">
      <code>efficient-kan</code> provides compact Kolmogorov-Arnold Network
      layers with explicit tensor semantics, adaptive-grid documentation,
      deterministic tests, local validation, and provenance tooling.
    </p>
    <div class="hero-actions">
      <a class="button primary" href="quickstart.html">Start with Quickstart</a>
      <a class="button" href="api.html">Read the API</a>
      <a class="button" href="https://github.com/shawcharles/efficient-kan">GitHub</a>
    </div>
  </div>
  <div class="hero-panel" aria-label="Package highlights">
    <div class="metric">
      <strong>Small public API</strong>
      <span><code>KANLinear</code>, <code>KAN</code>, and version metadata.</span>
    </div>
    <div class="metric">
      <strong>Minimal runtime dependency</strong>
      <span>The importable package depends on PyTorch.</span>
    </div>
    <div class="metric">
      <strong>Research-ready local checks</strong>
      <span>Tests, coverage, benchmark smoke, provenance, build, and wheel smoke.</span>
    </div>
  </div>
</section>

<section>
  <div class="section-heading">
    <h2>Start Here</h2>
    <p>Install the package, train the first model, and inspect the public API.</p>
  </div>
  <div class="card-grid">
    <article class="doc-card feature">
      <h3><a href="quickstart.html">Quickstart</a></h3>
      <p>
        Install commands and minimal examples for regression, binary
        classification, multiclass classification, and direct <code>KANLinear</code>
        use.
      </p>
    </article>
    <article class="doc-card feature">
      <h3><a href="api.html">API Reference</a></h3>
      <p>
        Public constructors, tensor shapes, return values, mutating methods,
        state tensors, and expected errors.
      </p>
    </article>
  </div>
</section>

<section>
  <div class="section-heading">
    <h2>Numerical Behavior</h2>
    <p>Document the modeling choices that matter for statistical use.</p>
  </div>
  <div class="card-grid">
    <article class="doc-card">
      <h3><a href="grid-updates.html">Grid Updates</a></h3>
      <p>
        How <code>update_grid()</code> mutates state, when to use it, when to
        avoid it, and how to handle degenerate batches.
      </p>
    </article>
    <article class="doc-card">
      <h3><a href="regularization.html">Regularization</a></h3>
      <p>
        The efficient spline-weight penalty, how it differs from
        activation-based KAN regularization, and objective examples.
      </p>
    </article>
    <article class="doc-card">
      <h3><a href="reproducibility.html">Reproducibility</a></h3>
      <p>
        Seeds, PyTorch version sensitivity, provenance commands, and downstream
        artifact expectations.
      </p>
    </article>
    <article class="doc-card">
      <h3><a href="benchmarking.html">Benchmarking</a></h3>
      <p>
        Local benchmark commands, measured operations, limitations, and
        same-machine comparison guidance.
      </p>
    </article>
  </div>
</section>

<section>
  <div class="section-heading">
    <h2>Maintainer Guides</h2>
    <p>Keep package changes auditable, local, and numerically explicit.</p>
  </div>
  <div class="card-grid">
    <article class="doc-card">
      <h3><a href="development.html">Development</a></h3>
      <p>
        Editable install, tests, linting, coverage, package constraints,
        dependency policy, and numerical-change process.
      </p>
    </article>
    <article class="doc-card">
      <h3><a href="release-checklist.html">Release Checklist</a></h3>
      <p>
        Local-only release gate, version updates, changelog expectations, build
        checks, wheel smoke tests, provenance, and the no-CI/CD assumption.
      </p>
    </article>
  </div>
</section>

<section class="callout">
  <h2>Local Validation</h2>
  <p>
    From the repository root, run the authoritative local readiness gate:
  </p>
  <pre><code>scripts/validate.sh</code></pre>
  <p>
    The repository intentionally uses local validation rather than hosted CI/CD.
  </p>
</section>
