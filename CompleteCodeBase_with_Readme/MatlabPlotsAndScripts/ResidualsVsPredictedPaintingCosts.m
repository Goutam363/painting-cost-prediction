y_test = y_test(:);
y_pred_lr = y_pred_lr(:);

if numel(y_test) ~= numel(y_pred_lr)
    error('Actual and predicted values must have equal lengths.');
end

valid = isfinite(y_test) & isfinite(y_pred_lr);

y_test = y_test(valid);
y_pred_lr = y_pred_lr(valid);

residuals = y_test - y_pred_lr;

predicted_lakh = y_pred_lr / 100000;
residuals_lakh = residuals / 100000;

assignin('base', 'residuals', residuals);
assignin('base', 'residuals_lakh', residuals_lakh);

figure('Color', 'w', 'Position', [100, 100, 900, 600]);

scatter(predicted_lakh, residuals_lakh, ...
    35, ...
    [0.30, 0.55, 0.72], ...
    'filled', ...
    'MarkerEdgeColor', [0.25, 0.25, 0.25]);

hold on;

xLimits = [min(predicted_lakh), max(predicted_lakh)];

plot(xLimits, [0, 0], ...
    '--', ...
    'Color', [0.85, 0.25, 0.15], ...
    'LineWidth', 2);

title('Residuals vs. Predicted Painting Costs — Linear Regression');

xlabel('Predicted Painting Cost (Lakh INR)');
ylabel('Residual (Lakh INR)');

legend('Test samples', 'Zero residual', ...
    'Location', 'best');

grid on;
box on;

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

xlim(xLimits);

set(gca, 'FontSize', 11);

hold off;

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/linear_regression_residuals.png', ...
    'Resolution', 300);

fprintf('Figure saved to figures/linear_regression_residuals.png\n');