# GITHUB_SETUP.md - GitHub Upload Instructions

## Complete Guide to Upload Your Repository to GitHub

This guide provides step-by-step instructions for uploading your DiffMOT + YOLOv10n repository to GitHub.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Git Configuration](#git-configuration)
3. [Create GitHub Repository](#create-github-repository)
4. [Initialize Local Repository](#initialize-local-repository)
5. [Push to GitHub](#push-to-github)
6. [Verify Upload](#verify-upload)
7. [Optional: Create Release](#optional-create-release)

---

## Prerequisites

### What You Need

- ✅ GitHub account (create at https://github.com/signup)
- ✅ Git installed on your system
- ✅ All documentation files in place
- ✅ Python source code ready
- ✅ .gitignore file present

### Check Git Installation

```bash
git --version
# Expected: git version 2.x.x or newer
```

If not installed:
```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Or download from: https://git-scm.com/downloads
```

---

## Git Configuration

### Step 1: Configure Git Identity

```bash
# Set your name (will appear in commits)
git config --global user.name "Your Name"

# Set your email (must match GitHub account)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 2: Setup Authentication

**Option A: HTTPS (Recommended for beginners)**

Use GitHub personal access token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Generate and **save the token** (you won't see it again)

**Option B: SSH (Recommended for advanced users)**

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/keys
```

---

## Create GitHub Repository

### Step 1: Go to GitHub

1. Visit: https://github.com/new
2. Or click the "+" icon → "New repository"

### Step 2: Configure Repository

**Repository Settings:**
- **Repository name**: `DiffMOT-YOLOv10n` (or your choice)
- **Description**: "Complete DiffMOT + YOLOv10n multi-object tracking pipeline"
- **Visibility**: 
  - ✅ Public (recommended - others can see and cite)
  - ⬜ Private (only you can access)
- **Initialize**:
  - ⬜ Do NOT add README (we already have one)
  - ⬜ Do NOT add .gitignore (we already have one)
  - ⬜ Do NOT add license yet (optional)

### Step 3: Click "Create Repository"

You'll see a page with setup instructions. **Keep this page open**.

---

## Initialize Local Repository

### Step 1: Navigate to Your Project

```bash
cd /path/to/DIFFMOT_Upgrade
```

### Step 2: Verify Files Are Present

```bash
# Check essential files
ls -la

# Expected output should include:
# README.md
# SETUP.md
# PIPELINE.md
# REFERENCES.md
# .gitignore
# main.py
# configs/
# (and other files)
```

### Step 3: Initialize Git Repository

```bash
# Initialize git
git init

# Check status
git status
```

### Step 4: Add Files to Git

```bash
# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status

# Expected: All .md, .py, .yaml files staged
# Expected: DanceTrack/, cache/, results/ NOT staged (in .gitignore)
```

### Step 5: Create Initial Commit

```bash
git commit -m "Initial commit: Complete DiffMOT + YOLOv10n pipeline with comprehensive documentation"

# Verify commit
git log --oneline
```

---

## Push to GitHub

### Step 1: Add Remote Repository

**For HTTPS:**
```bash
git remote add origin https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n.git

# Replace YOUR_USERNAME with your GitHub username
```

**For SSH:**
```bash
git remote add origin git@github.com:YOUR_USERNAME/DiffMOT-YOLOv10n.git
```

### Step 2: Verify Remote

```bash
git remote -v

# Expected output:
# origin  https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n.git (fetch)
# origin  https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n.git (push)
```

### Step 3: Push to GitHub

```bash
# Set default branch name (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**For HTTPS**: Enter your GitHub username and **personal access token** (not password)

**For SSH**: Should work automatically if SSH key is configured

### Step 4: Verify Upload

```bash
# Check push status
git status

# Expected: "Your branch is up to date with 'origin/main'"
```

---

## Verify Upload

### Step 1: Visit GitHub Repository

Go to: `https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n`

### Step 2: Verify Files

Check that you see:
- ✅ README.md displayed on main page
- ✅ All .md files in file list
- ✅ All .py files
- ✅ configs/ directory
- ✅ .gitignore file
- ✅ No DanceTrack/ directory (excluded by .gitignore)
- ✅ No cache/ or results/ (excluded by .gitignore)

### Step 3: Check README Display

- ✅ README.md renders properly
- ✅ DiffMOT badge shows at top
- ✅ Links work
- ✅ Formatting correct

### Step 4: Check Navigation

Click on:
- ✅ SETUP.md link → Opens setup guide
- ✅ PIPELINE.md link → Opens technical docs
- ✅ REFERENCES.md link → Opens citations

---

## Optional: Create Release

### Step 1: Create a Tag

```bash
# Create annotated tag
git tag -a v1.0 -m "Version 1.0: Complete DiffMOT + YOLOv10n implementation"

# Push tag to GitHub
git push origin v1.0
```

### Step 2: Create Release on GitHub

1. Go to repository page
2. Click "Releases" → "Create a new release"
3. Select tag: `v1.0`
4. Release title: `v1.0 - Initial Release`
5. Description:
   ```
   # DiffMOT + YOLOv10n v1.0
   
   Complete multi-object tracking pipeline implementation.
   
   ## Features
   - YOLOv10n detection
   - FastReID feature extraction
   - DiffMOT tracking with D2MP
   - Complete documentation
   - Training and inference pipelines
   
   ## Citation
   Based on DiffMOT: https://github.com/Kroery/DiffMOT
   
   See REFERENCES.md for complete citations.
   ```
6. Click "Publish release"

---

## Adding Repository Badges

### Step 1: Edit README.md

Add badges at the top (already included in README.md):

```markdown
[![GitHub](https://img.shields.io/badge/GitHub-YourUsername/DiffMOT--YOLOv10n-blue?style=flat-square)](https://github.com/YourUsername/DiffMOT-YOLOv10n)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
```

### Step 2: Commit and Push

```bash
git add README.md
git commit -m "Update README with repository badge"
git push
```

---

## Repository Settings (Optional)

### Step 1: Go to Repository Settings

Visit: `https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n/settings`

### Step 2: Configure Options

**General:**
- ✅ Enable issues (for bug reports)
- ✅ Enable discussions (optional)
- ⬜ Disable wiki (optional)

**Branches:**
- Set `main` as default branch
- (Optional) Add branch protection rules

**Topics:**
Add relevant tags:
- `multi-object-tracking`
- `diffusion-models`
- `yolov10`
- `computer-vision`
- `pytorch`
- `dancetrack`

---

## Making Updates

### After Initial Upload

```bash
# 1. Make changes to files
# 2. Check status
git status

# 3. Add changes
git add .

# 4. Commit with descriptive message
git commit -m "Update documentation: Add troubleshooting section"

# 5. Push to GitHub
git push
```

---

## Troubleshooting

### Issue: Authentication Failed (HTTPS)

**Solution**: Use personal access token, not password
1. Generate token: https://github.com/settings/tokens
2. Use token as password when pushing

### Issue: SSH Key Not Working

**Solution**: Check SSH key is added to GitHub
```bash
# Test SSH connection
ssh -T git@github.com

# Expected: "Hi username! You've successfully authenticated"
```

### Issue: Large Files Rejected

**Solution**: Ensure .gitignore excludes large files
```bash
# Check .gitignore includes:
DanceTrack/
*.pt
*.pth
weights/
```

### Issue: Remote Already Exists

**Solution**: Remove and re-add remote
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n.git
```

---

## Sharing Your Repository

### Share Links

**Main repository**: 
```
https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n
```

**Specific file**: 
```
https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n/blob/main/SETUP.md
```

**Clone URL** (for others):
```bash
git clone https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n.git
```

### Social Media

Share on:
- Twitter/X with hashtags: #ComputerVision #MultiObjectTracking #PyTorch
- LinkedIn with project description
- Research communities

### Academic Citation

Include GitHub link in papers:
```latex
\footnote{Code available at: \url{https://github.com/YOUR_USERNAME/DiffMOT-YOLOv10n}}
```

---

## ✅ Success Checklist

Before sharing publicly:

- [ ] All files pushed to GitHub
- [ ] README.md displays correctly
- [ ] All links work
- [ ] DiffMOT properly cited
- [ ] .gitignore working (no large files)
- [ ] License file added (optional)
- [ ] Repository description set
- [ ] Topics/tags added
- [ ] Release created (optional)
- [ ] Tested clone from GitHub

---

## Next Steps

1. ✅ Star the original DiffMOT repository
2. ✅ Share your implementation
3. ✅ Receive feedback and improve
4. ✅ Contribute back to community

---

**Status**: ✅ Ready to upload to GitHub  
**Last Updated**: January 7, 2026

For more information, see [README.md](README.md) and [PRE_GITHUB_CHECKLIST.md](PRE_GITHUB_CHECKLIST.md)
