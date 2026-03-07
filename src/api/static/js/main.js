// src/api/static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('queryForm');
    const questionInput = document.getElementById('questionInput');
    const submitBtn = document.getElementById('submitBtn');
    const resultDiv = document.getElementById('result');

    if (!form || !questionInput || !submitBtn || !resultDiv) {
        console.error('Required elements not found in DOM');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();

        if (!question) {
            alert('请输入您的问题');
            return;
        }

        // 更新按钮状态
        submitBtn.disabled = true;
        const originalText = submitBtn.textContent;
        submitBtn.textContent = '⏳ 处理中...';

        // 清空之前的结果
        resultDiv.innerHTML = '';

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (response.ok) {
                // 安全地渲染答案（防止 XSS）
                let answerHtml = escapeHtml(data.answer);

                // 将 <Source: DB00945> 转为可点击链接
                answerHtml = answerHtml.replace(
                    /&lt;Source:\s*([A-Za-z0-9_]+)&gt;/g,
                    '<a href="https://go.drugbank.com/drugs/$1" target="_blank" rel="noopener noreferrer" class="citation">&lt;Source: $1&gt;</a>'
                );

                resultDiv.innerHTML = `
                    <div class="card">
                        <h3>💡 回答</h3>
                        <div class="answer">${answerHtml}</div>
                    </div>
                `;
            } else {
                throw new Error(data.detail || '服务器返回错误');
            }
        } catch (error) {
            console.error('Query error:', error);
            resultDiv.innerHTML = `
                <div class="error">
                    ❌ 查询失败: ${escapeHtml(error.message)}
                </div>
            `;
        } finally {
            // 恢复按钮
            submitBtn.disabled = false;
            submitBtn.textContent = originalText;
        }
    });

    // 简单的 HTML 转义函数（防 XSS）
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});