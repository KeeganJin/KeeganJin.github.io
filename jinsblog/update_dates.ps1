# PowerShell脚本：批量更新文章日期
# 按照学习顺序将文章分散在 2025年10月 到 2026年5月

$dateMap = @{
    # ===== 2025年10月：数学基础和入门 =====
    'linear-algebra' = '2025-10-01'
    'probability-theory' = '2025-10-05'
    'statistics' = '2025-10-08'
    'ml-introduction' = '2025-10-12'

    # ===== 2025年10月中：传统ML算法 =====
    'linear-regression' = '2025-10-15'
    'logistic-regression' = '2025-10-18'
    'knn' = '2025-10-21'
    'bayesian-classifier' = '2025-10-24'
    'svm' = '2025-10-27'

    # ===== 2025年11月：数据处理和决策树 =====
    'data-preprocessing' = '2025-11-01'
    'feature-engineering' = '2025-11-04'
    'decision-tree' = '2025-11-07'
    'ensemble-learning' = '2025-11-10'

    # ===== 2025年11月中：Boosting系列 =====
    'boosting' = '2025-11-13'
    'random-forest' = '2025-11-16'
    'gbdt' = '2025-11-19'
    'xgboost' = '2025-11-22'
    'xgboost算法' = '2025-11-23'
    'lightgbm' = '2025-11-25'
    'stacking-blending' = '2025-11-28'

    # ===== 2025年12月：神经网络基础 =====
    'neural-network-intro' = '2025-12-01'
    'activation-functions' = '2025-12-04'
    'backpropagation' = '2025-12-07'
    'loss-functions' = '2025-12-10'
    'regularization' = '2025-12-13'
    'optimization' = '2025-12-16'
    'optimization-algorithms' = '2025-12-17'

    # ===== 2025年12月中：深度学习架构 =====
    'cnn' = '2025-12-20'
    'rnn' = '2025-12-23'
    'lstm-gru' = '2025-12-26'

    # ===== 2026年1月：注意力机制和Transformer =====
    'attention-mechanism' = '2026-01-01'
    'LM-1-transformer' = '2026-01-05'

    # ===== 2026年1月中：生成模型 =====
    'gan' = '2026-01-08'
    'vae' = '2026-01-11'
    'diffusion-models' = '2026-01-14'

    # ===== 2026年1月下旬：强化学习 =====
    'rl-introduction' = '2026-01-17'
    'rl-value-based' = '2026-01-20'
    'rl-policy-based' = '2026-01-23'
    'rl-advanced' = '2026-01-26'

    # ===== 2026年2月：大模型架构 =====
    'llm-architecture' = '2026-02-01'
    'LM-2-模型架构' = '2026-02-05'
    'pretraining-lm' = '2026-02-08'
    'fine-tuning' = '2026-02-11'
    'prompt-engineering' = '2026-02-14'

    # ===== 2026年2月中：训练和部署 =====
    'distributed-training' = '2026-02-17'
    'model-compression' = '2026-02-20'
    'model-deployment' = '2026-02-23'

    # ===== 2026年2月下旬：Agent入门 =====
    'agent-intro' = '2026-02-26'
    'llm-and-agent' = '2026-02-28'

    # ===== 2026年3月：Agent架构 =====
    'agent-architecture' = '2026-03-01'
    'tool-use' = '2026-03-04'
    'agent-memory' = '2026-03-07'
    'agent-planning' = '2026-03-10'

    # ===== 2026年3月中：Agent框架 =====
    'agent-frameworks' = '2026-03-13'
    'langchain-agent' = '2026-03-16'
    'autogen-agent' = '2026-03-19'
    'crewai-agent' = '2026-03-22'
    'semantic-kernel' = '2026-03-25'

    # ===== 2026年3月下旬：多Agent =====
    'multi-agent-intro' = '2026-03-28'
    'multi-agent-collaboration' = '2026-03-30'
    'multi-agent-competition' = '2026-04-01'
    'hierarchical-agent' = '2026-04-04'

    # ===== 2026年4月：Agent应用 =====
    'code-assistant-agent' = '2026-04-07'
    'data-analysis-agent' = '2026-04-10'
    'customer-service-agent' = '2026-04-13'
    'research-agent' = '2026-04-16'
    'agent-software-engineering' = '2026-04-19'

    # ===== 2026年4月中：Agent工程化 =====
    'agent-debugging' = '2026-04-22'
    'agent-security' = '2026-04-25'
    'agent-evaluation' = '2026-04-28'
    'agent-ethics' = '2026-05-01'
    'OpenClaw拆解' = '2026-05-04'

    # ===== 2026年5月：前沿研究 =====
    'agent-future' = '2026-05-07'
    '多模态' = '2026-05-10'

    # ===== 机器学习综合文章 =====
    '树模型' = '2026-05-13'

    # ===== 随笔 =====
    '人生的奥德赛' = '2026-05-15'
    'blog-redesign' = '2026-05-17'
}

$basePath = "source\_posts\zh-CN"
$updatedCount = 0

foreach ($category in @("技术文档\机器学习", "技术文档\Agent", "随笔")) {
    $categoryPath = Join-Path $basePath $category
    if (Test-Path $categoryPath) {
        Get-ChildItem -Path $categoryPath -Filter "*.md" | ForEach-Object {
            $fileName = $_.BaseName
            if ($dateMap.ContainsKey($fileName)) {
                $newDate = $dateMap[$fileName]
                $content = Get-Content $_.FullName -Raw
                $pattern = "date:\s*\d{4}-\d{2}-\d{2}"
                $newLine = "date: $newDate"

                if ($content -match $pattern) {
                    $newContent = $content -replace $pattern, $newLine
                    Set-Content -Path $_.FullName -Value $newContent -NoNewline
                    Write-Host "Updated: $fileName -> $newDate"
                    $updatedCount++
                }
            }
        }
    }
}

Write-Host "`n总计更新 $updatedCount 篇文章日期"