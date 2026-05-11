/**
 * Copy post images to correct output location
 *
 * Markdown uses: assets/images/${docName}/${imageFile}
 * HTML outputs: /assets/images/${docName}/${imageFile}
 *
 * This script copies images from source posts to docs/assets/images/
 */

const fs = require('fs');
const path = require('path');

hexo.on('generateAfter', function() {
  const docsDir = path.join(hexo.base_dir, '..', 'docs');
  const assetsDir = path.join(docsDir, 'assets', 'images');

  // Ensure output directory exists
  if (!fs.existsSync(assetsDir)) {
    fs.mkdirSync(assetsDir, { recursive: true });
  }

  // Copy Chinese 树模型 images
  const zhSourceDir = path.join(hexo.source_dir, '_posts', 'zh-CN', '技术文档', '机器学习', 'assets', 'images', '树模型');
  const zhTargetDir = path.join(assetsDir, '树模型');

  if (fs.existsSync(zhSourceDir)) {
    if (!fs.existsSync(zhTargetDir)) {
      fs.mkdirSync(zhTargetDir, { recursive: true });
    }

    const files = fs.readdirSync(zhSourceDir);
    files.forEach(file => {
      const src = path.join(zhSourceDir, file);
      const dest = path.join(zhTargetDir, file);
      fs.copyFileSync(src, dest);
    });
    hexo.log.info('Copied 树模型 images to output');
  }

  // Copy English tree-models images
  const enSourceDir = path.join(hexo.source_dir, '_posts', 'en', 'tech-docs', 'machine-learning', 'assets', 'images', 'tree-models');
  const enTargetDir = path.join(assetsDir, 'tree-models');

  if (fs.existsSync(enSourceDir)) {
    if (!fs.existsSync(enTargetDir)) {
      fs.mkdirSync(enTargetDir, { recursive: true });
    }

    const files = fs.readdirSync(enSourceDir);
    files.forEach(file => {
      const src = path.join(enSourceDir, file);
      const dest = path.join(enTargetDir, file);
      fs.copyFileSync(src, dest);
    });
    hexo.log.info('Copied tree-models images to output');
  }

  // Scan all other assets/images directories in posts
  const postsDir = path.join(hexo.source_dir, '_posts');

  function scanAndCopy(dir) {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    items.forEach(item => {
      const fullPath = path.join(dir, item.name);

      if (item.isDirectory()) {
        // Check if this is an assets/images directory
        if (item.name === 'assets') {
          const imagesDir = path.join(fullPath, 'images');
          if (fs.existsSync(imagesDir)) {
            // Copy all subdirectories in images/
            const imageSubDirs = fs.readdirSync(imagesDir, { withFileTypes: true });
            imageSubDirs.forEach(subDir => {
              if (subDir.isDirectory()) {
                const sourceSubDir = path.join(imagesDir, subDir.name);
                const targetSubDir = path.join(assetsDir, subDir.name);

                if (!fs.existsSync(targetSubDir)) {
                  fs.mkdirSync(targetSubDir, { recursive: true });
                }

                const pngFiles = fs.readdirSync(sourceSubDir);
                pngFiles.forEach(pngFile => {
                  const src = path.join(sourceSubDir, pngFile);
                  const dest = path.join(targetSubDir, pngFile);
                  if (fs.statSync(src).isFile()) {
                    fs.copyFileSync(src, dest);
                  }
                });
                hexo.log.info(`Copied ${subDir.name} images to output`);
              }
            });
          }
        } else {
          // Continue scanning subdirectories
          scanAndCopy(fullPath);
        }
      }
    });
  }

  scanAndCopy(postsDir);
});