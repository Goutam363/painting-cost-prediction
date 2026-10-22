models = {'LR', 'DT'};

mae_inr = [4443792.31, 5227013.59];

rmse_inr = [6862920.46, 10253604.58];

mae_lakh = mae_inr / 100000;
rmse_lakh = rmse_inr / 100000;

assignin('base', 'models', models);
assignin('base', 'mae_inr', mae_inr);
assignin('base', 'rmse_inr', rmse_inr);
assignin('base', 'mae_lakh', mae_lakh);
assignin('base', 'rmse_lakh', rmse_lakh);

figure('Color', 'w', 'Position', [100, 100, 900, 550]);

values = [mae_lakh(:), rmse_lakh(:)];

b = bar(values, 'grouped');

b(1).FaceColor = [0.12, 0.47, 0.71];
b(2).FaceColor = [1.00, 0.45, 0.00];

for k = 1:numel(b)
    x = b(k).XEndPoints;
    y = b(k).YEndPoints;
    labels = compose('%.2f', y);

    text(x, y, labels, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 10);
end

title('Test Error Comparison of Regression Models');
xlabel('Machine Learning Model');
ylabel('Error (Lakh INR)');

xticks(1:numel(models));
xticklabels(models);

legend({'MAE', 'RMSE'}, 'Location', 'northwest');

grid on;
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

ylim([0, max(rmse_lakh) * 1.18]);

set(gca, 'FontSize', 11);

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/test_error_comparison.png', ...
    'Resolution', 300);

fprintf('Figure saved to figures/test_error_comparison.png\n');