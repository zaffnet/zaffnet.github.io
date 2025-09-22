# 1) Update _config.yml: rename gems->plugins and exclude vendor
sed -i.bak 's/^\s*gems\s*:/plugins:/g' _config.yml

grep -q '^exclude:' _config.yml || cat >> _config.yml <<'YAML'
exclude:
  - vendor/
  - vendor/**/*
  - .bundle/
  - node_modules/
  - Gemfile
  - Gemfile.lock
YAML

# 2) Pin rubyzip < 3 to avoid breaking API, add webrick and faraday-retry
grep -q 'rubyzip' Gemfile || echo 'gem "rubyzip", "~> 2.3"' >> Gemfile
grep -q 'webrick' Gemfile || echo 'gem "webrick", "~> 1.8"' >> Gemfile
grep -q 'faraday-retry' Gemfile || echo 'gem "faraday-retry", "~> 2.2"' >> Gemfile

# 3) Install/update
bundle config set path vendor/bundle
bundle install || bundle update

# 4) Clean and serve
rm -rf _site .jekyll-cache .jekyll-metadata
bundle exec jekyll serve --watch --livereload --force_polling

