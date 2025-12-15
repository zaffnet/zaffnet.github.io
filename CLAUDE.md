# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based personal website and blog hosted on GitHub Pages at https://zafarmahmood.com. The site showcases technical writing, publications, projects, and professional experience in NLP and AI engineering.

## Build and Development Commands

### Local Development
```bash
# Install dependencies (Ruby 3.4+ required)
bundle install

# Serve locally with auto-rebuild
bundle exec jekyll serve

# Build site to _site/ directory
bundle exec jekyll build

# Clean build artifacts
bundle exec jekyll clean
```

### Testing
```bash
# Check for broken links, HTML validation
bundle exec jekyll build && htmlproofer ./_site --disable-external

# Validate YAML front matter
ruby -ryaml -e "YAML.load_file('_posts/YYYY-MM-DD-post-name.md')"
```

## Architecture

### Core Structure
- **Jekyll 4.4** with Minima theme (customized)
- **Content**: Blog posts in `_posts/` with YAML front matter
- **Layouts**: Custom layouts in `_layouts/` (default.html, post.html)
- **Styling**: SCSS in `_sass/minima/` + custom CSS in `assets/css/`
- **Pages**: Main portfolio (index.html), blog listing (blogs.html)

### Key Plugins
- `jekyll-seo-tag`: Auto-generates meta tags for SEO
- `jekyll-feed`: Creates RSS/Atom feed at /feed.xml
- `jekyll-sitemap`: Auto-generates sitemap.xml
- `jekyll-mentions`: Enables @username syntax
- `jekyll-redirect-from`: Handles URL redirects
- `jekyll-gist`: Embeds GitHub Gists

### Blog Post Format
Posts in `_posts/` must follow naming: `YYYY-MM-DD-title.md`

Required front matter:
```yaml
---
layout: post
title: "Post Title"
desc: "Brief description for SEO"
keywords: "comma, separated, keywords"
date: YYYY-MM-DD HH:MM:SS +0000
lastmod: YYYY-MM-DD HH:MM:SS +0000
comments: true
permalink: custom-url-slug
---
```

### Custom Features
- **Hand-drawn SVG diagrams**: Blog posts include custom Excalidraw-style SVG graphics in `/assets/img/`
- **Skills showcase**: Interactive skill badges with hover effects (index.html lines 104-278)
- **Timeline layout**: Professional experience timeline with visual markers
- **Custom SCSS variables**: Defined in `_sass/minima.scss` (colors, spacing, responsive breakpoints)

## GitHub Actions Workflow

`.github/workflows/main.yml` automatically:
1. Merges `main` into all feature branches on every push to main
2. Uses git worktrees to handle multiple branches safely
3. Skips branches with merge conflicts and logs them

This keeps feature branches updated with main automatically.

## Content Guidelines

### Blog Posts
- Use natural, conversational tone (see existing posts for voice)
- Include hand-drawn SVG diagrams for visual concepts
- Keep code examples concise with inline comments
- Add descriptive alt text for images
- Date format in filenames must match format in front matter

### Visual Assets
- SVG diagrams: Store in `/assets/img/` with descriptive names
- Skill icons: PNG/SVG in `/assets/img/skills/`
- Profile images: `/assets/` root directory
- Use lazy loading: `loading="lazy"` for images below fold

## Configuration Notes

### _config.yml
- **Timezone**: America/Toronto
- **Permalink structure**: `/:categories/:title/`
- **Markdown**: Kramdown with GitHub Flavored Markdown (GFM)
- **Excluded from build**: vendor/, .bundle/, node_modules/, .jekyll-cache/

### Sass Configuration
- Uses `sass-embedded` (Ruby Sass is deprecated)
- SCSS partials imported in `_sass/minima.scss`
- Custom styles in `assets/css/style.css`
- Syntax highlighting in `assets/css/syntax.css`

## Common Pitfalls

1. **Blog post dates**: Posts with future dates won't appear on the site
2. **YAML front matter**: Ensure proper indentation and valid YAML syntax
3. **Asset paths**: Use `{{ "assets/path/to/file" | relative_url }}` in templates
4. **SVG files**: Ensure xmlns attribute is present for proper rendering
5. **Bundle updates**: Run `bundle update` cautiously; lock tested versions in Gemfile.lock

## Deployment

Site is automatically deployed via GitHub Pages from the `main` branch. No manual deployment steps required. Changes pushed to main are live within 1-2 minutes.
