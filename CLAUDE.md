# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal portfolio and blog site for Zafarullah Mahmood, built with Jekyll and hosted on GitHub Pages at zafarmahmood.com.

## Build & Serve

**Prerequisites**: Ruby 3.4+ (pinned in `.ruby-version`) and Bundler 2.x. Install Ruby via [rbenv](https://github.com/rbenv/rbenv) or [mise](https://mise.jdx.dev/) — do not use macOS system Ruby.

```bash
# Install dependencies (use vendor/bundle to isolate from system gems)
bundle config set --local path vendor/bundle
MAKEFLAGS="CXX=clang++" bundle install

# Serve locally with live reload at http://localhost:4000
bundle exec jekyll serve --livereload
```

The site builds to `_site/` (gitignored).

### Common installation issues

- **`sass-embedded` or `google-protobuf` fails**: These ship platform-specific native binaries. Run `bundle lock --add-platform arm64-darwin` (Apple Silicon) or `x86_64-linux-gnu` (Linux), then `bundle install` again.
- **`nokogiri` build fails**: Needs libxml2/libxslt. On macOS: `brew install libxml2 libxslt`. Usually the pre-built native gem works without this.
- **`eventmachine` build fails (CXX=false)**: Ruby 3.4.8 via rbenv may be compiled with `CXX=false`, causing eventmachine's native extension to fail (`make: *** [binder.o] Error 1`). Fix: `MAKEFLAGS="CXX=clang++" bundle install`. The MAKEFLAGS prefix is only needed during `bundle install`; `bundle exec jekyll serve` works without it once gems are compiled.
- **Wrong Ruby version**: If you see `Your Ruby version is X but your Gemfile specified ~> 3.4`, install Ruby 3.4 via `rbenv install 3.4.8` (or whichever patch). The `.ruby-version` file will auto-select it.
- **Port 4000 in use**: Use `bundle exec jekyll serve --port 4001`.
- **Sass deprecation warnings**: Already suppressed via `sass: quiet_deps: true` in `_config.yml`.

## Architecture

- **Jekyll static site** using the `minima` theme with local SCSS overrides in `_sass/minima/`
- **Layouts**: `_layouts/default.html` (base, standalone HTML5 document with inline nav) → `_layouts/post.html` (blog posts)
- **Pages**: `index.html` (homepage with hero, about, timeline), `blogs.html` (post listing), `404.html`
- **Blog posts**: Markdown files in `_posts/` using Kramdown with GFM input
- **Styling**: `assets/css/style.css` (main) and `assets/css/syntax.css` (code highlighting); uses CSS custom properties for theming (accent: `#007aff`); Plus Jakarta Sans via Google Fonts
- **Client-side JS**: MathJax for math rendering, Mermaid v11 (ESM) for diagrams, Font Awesome for icons — all loaded in the default layout
- **Images**: `assets/img/skills/` (tech skill icons), `assets/img/archive/` (article diagrams)
- **Includes**: `_includes/header.html` and `_includes/footer.html` exist but are legacy/unused — the default layout is self-contained

## Blog Post Conventions

Posts use this frontmatter structure:

```yaml
layout: post
title: "Title"
desc: "Short description"
keywords: "comma, separated, keywords"
date: YYYY-MM-DD 00:00:00 +0000
lastmod: YYYY-MM-DD 00:00:00 +0000
comments: true
permalink: slug-without-slashes
```

Posts support Mermaid diagram blocks (` ```mermaid `) and MathJax notation. The writing style is conversational and uses humor.

## CI/CD

A GitHub Actions workflow (`.github/workflows/main.yml`) automatically merges `main` into all feature branches on every push to `main`. It uses git worktrees to handle branch names with slashes and skips branches with merge conflicts.

## Permalink Structure

Configured as `/:categories/:title/` in `_config.yml`. Blog posts typically set a custom `permalink` in frontmatter.
