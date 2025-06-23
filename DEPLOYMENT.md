# GitHub Pages Deployment Guide

This guide explains how to deploy the Borgia framework documentation site to GitHub Pages.

## Prerequisites

- GitHub repository with the Borgia framework code
- GitHub account with Pages enabled
- Ruby and Jekyll installed locally (for development)

## Setup Instructions

### 1. Repository Configuration

1. **Enable GitHub Pages**:
   - Go to your repository settings
   - Scroll down to "Pages" section
   - Under "Source", select "GitHub Actions"

2. **Configure Repository Settings**:
   - Ensure the repository is public (required for free GitHub Pages)
   - Or have GitHub Pro/Team for private repository Pages

### 2. Domain Configuration (Optional)

If you want to use a custom domain:

1. **Purchase a domain** from a domain registrar
2. **Configure DNS**:
   - Add a CNAME record pointing to `your-username.github.io`
   - Or add A records pointing to GitHub's IP addresses:
     - 185.199.108.153
     - 185.199.109.153
     - 185.199.110.153
     - 185.199.111.153

3. **Update the workflow**:
   - Edit `.github/workflows/github-pages.yml`
   - Replace `your-custom-domain.com` with your actual domain
   - Or remove the `cname` line to use the default GitHub Pages URL

### 3. Local Development

To develop and test the site locally:

```bash
# Install Ruby dependencies
bundle install

# Serve the site locally
bundle exec jekyll serve

# Open http://localhost:4000 in your browser
```

### 4. Content Updates

The site is automatically rebuilt when you push changes to the main branch. The content is organized as follows:

```
├── _config.yml           # Jekyll configuration
├── _includes/
│   └── head.html         # Custom head with MathJax
├── assets/
│   └── css/
│       └── custom.css    # Custom styling
├── index.md              # Homepage
├── theoretical-foundations.md
├── implementation.md
├── api-reference.md
├── examples.md
├── publications.md
├── Gemfile               # Ruby dependencies
└── .github/
    └── workflows/
        └── github-pages.yml  # Deployment workflow
```

### 5. Customization

#### Updating Site Information

Edit `_config.yml` to customize:
- Site title and description
- Base URL and repository URL
- Navigation menu items
- Plugin configuration

#### Styling

Modify `assets/css/custom.css` to customize:
- Color scheme
- Typography
- Layout and spacing
- Responsive design

#### Mathematical Equations

The site supports LaTeX math notation via MathJax:
- Inline math: `$equation$` or `\(equation\)`
- Display math: `$$equation$$` or `\[equation\]`

#### Code Highlighting

Code blocks are automatically highlighted using Rouge:
```rust
// Rust code example
fn main() {
    println!("Hello, Borgia!");
}
```

### 6. Deployment Process

The deployment is automated via GitHub Actions:

1. **Trigger**: Push to main branch or pull request
2. **Build**: Jekyll builds the static site
3. **Test**: HTMLProofer validates the generated HTML
4. **Deploy**: Site is deployed to GitHub Pages (main branch only)

### 7. Monitoring and Analytics

#### Built-in Analytics

GitHub Pages provides basic analytics in your repository's Insights tab.

#### Google Analytics (Optional)

To add Google Analytics:

1. Create a Google Analytics account
2. Add your tracking ID to `_config.yml`:
   ```yaml
   google_analytics: UA-XXXXXXXXX-X
   ```

#### Custom Analytics

You can add custom analytics by modifying `_includes/head.html`.

### 8. SEO Optimization

The site includes SEO optimization via:
- Jekyll SEO Tag plugin
- Proper meta tags and Open Graph data
- Sitemap generation
- RSS feed

### 9. Performance Optimization

#### Image Optimization

- Use WebP format when possible
- Compress images before adding to repository
- Consider using a CDN for large assets

#### Caching

GitHub Pages automatically handles caching headers for optimal performance.

### 10. Troubleshooting

#### Common Issues

1. **Build Failures**:
   - Check the Actions tab for error messages
   - Verify Jekyll configuration syntax
   - Ensure all dependencies are properly specified

2. **404 Errors**:
   - Check file paths and permalinks
   - Verify baseurl configuration
   - Ensure files are committed to the repository

3. **Styling Issues**:
   - Clear browser cache
   - Check CSS file paths
   - Verify custom CSS syntax

4. **Math Rendering**:
   - Ensure MathJax is properly loaded
   - Check equation syntax
   - Verify browser JavaScript is enabled

#### Debug Mode

To debug locally with verbose output:
```bash
bundle exec jekyll serve --verbose --trace
```

### 11. Security Considerations

- Keep dependencies updated
- Review third-party plugins for security issues
- Use HTTPS (automatically enabled on GitHub Pages)
- Be cautious with user-generated content

### 12. Backup and Version Control

- All content is version-controlled in Git
- GitHub automatically maintains backups
- Consider periodic exports for additional backup

## Support

For issues with:
- **Jekyll**: Check the [Jekyll documentation](https://jekyllrb.com/docs/)
- **GitHub Pages**: See [GitHub Pages documentation](https://docs.github.com/en/pages)
- **Borgia Framework**: Open an issue in the repository

## Example URLs

After deployment, your documentation will be available at:
- Default: `https://your-username.github.io/borgia/`
- Custom domain: `https://your-custom-domain.com/`

Individual pages:
- Theoretical Foundations: `/theoretical-foundations/`
- Implementation: `/implementation/`
- API Reference: `/api-reference/`
- Examples: `/examples/`
- Publications: `/publications/`

---

*This deployment guide ensures your Borgia framework documentation is professionally presented and easily accessible to the scientific community.* 