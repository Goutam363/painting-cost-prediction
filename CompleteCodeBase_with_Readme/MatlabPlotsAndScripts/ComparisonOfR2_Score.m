models = {'LR', 'DT'};

r2_scores = [0.9441, 0.8753];

assignin('base', 'models', models);
assignin('base', 'r2_scores', r2_scores);

figure('Color', 'w', 'Position', [100, 100, 900, 500]);

bars = bar(r2_scores, ...
    0.6, ...
    'FaceColor', [0.12, 0.47, 0.71], ...
    'EdgeColor', 'black');

xtips = bars.XEndPoints;
ytips = bars.YEndPoints;

labels = compose('%.4f', r2_scores);

text(xtips, ytips, labels, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 11);

title('Comparison of Test R^2 Scores');
xlabel('Machine Learning Model');
ylabel('Test R^2 Score');

xticks(1:numel(models));
xticklabels(models);

ylim([0, 1.05]);

grid on;

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

set(gca, 'FontSize', 11);

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/test_r2_comparison.png', ...
    'Resolution', 300);

fprintf('Figure saved to figures/test_r2_comparison.png\n');