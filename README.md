---
layout: post
title: "README"
date:   2017-01-10 15:50:06 +0530
title: README
permalink: /readme
sitemap:
    priority: 1.0
    changefreq: 'monthly'
    lastmod: 2017-01-06 15:50:06 +0530
---
# [zaffnet.github.io](https://zaffnet.github.io)

[![pages-build-deployment](https://github.com/zaffnet/zaffnet.github.io/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/zaffnet/zaffnet.github.io/actions/workflows/pages/pages-build-deployment)
## Local development with Docker

If you prefer not to install Ruby, Bundler, or Jekyll locally, you can use the Docker helper script:

```bash
./build.sh
```

The script cleans previous build caches and starts a containerised Jekyll server on http://localhost:4000, mounting the current directory so that your latest changes are served immediately.
