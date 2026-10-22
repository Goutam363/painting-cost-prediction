y_test = y_test(:);
y_pred_lr = y_pred_lr(:);

if numel(y_test) ~= numel(y_pred_lr)
    error('Actual and predicted values must have equal lengths.');
end

valid = isfinite(y_test) & isfinite(y_pred_lr);

y_test = y_test(valid);
y_pred_lr = y_pred_lr(valid);

actual_lakh = y_test / 100000;
predicted_lakh = y_pred_lr / 100000;

assignin('base', 'actual_lakh', actual_lakh);
assignin('base', 'predicted_lakh', predicted_lakh);

figure('Color', 'w', 'Position', [100, 100, 900, 600]);

scatter(actual_lakh, predicted_lakh, ...
    35, ...
    [0.30, 0.55, 0.72], ...
    'filled', ...
    'MarkerEdgeColor', [0.25, 0.25, 0.25], ...
    'DisplayName', 'Test samples');

hold on;

minVal = min([actual_lakh; predicted_lakh]);
maxVal = max([actual_lakh; predicted_lakh]);

plot([minVal, maxVal], [minVal, maxVal], ...
    '--', ...
    'Color', [0.10, 0.45, 0.75], ...
    'LineWidth', 2, ...
    'DisplayName', 'Perfect prediction');

title('Actual vs. Predicted Painting Costs — Linear Regression');

xlabel('Actual Painting Cost (Lakh INR)');
ylabel('Predicted Painting Cost (Lakh INR)');

legend('Location', 'northwest');

grid on;
box on;

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

xlim([min(0, minVal), maxVal * 1.05]);
ylim([min(0, minVal), maxVal * 1.05]);

set(gca, 'FontSize', 11);

hold off;

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/linear_regression_actual_vs_predicted.png', ...
    'Resolution', 300);

fprintf('Figure saved to figures/linear_regression_actual_vs_predicted.png\n');