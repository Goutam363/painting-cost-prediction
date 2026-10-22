paintingCost = paintingCost(:);

paintingCost = paintingCost( ...
    isfinite(paintingCost) & paintingCost > 0);

figure('Color', 'w', 'Position', [100, 100, 900, 500]);

histogram(paintingCost, ...
    'Normalization', 'pdf', ...
    'NumBins', 30, ...
    'FaceColor', [0.40, 0.63, 0.78], ...
    'EdgeColor', [0.25, 0.25, 0.25], ...
    'LineWidth', 0.8);

hold on;

[xKDE, yKDE] = ksdensity(paintingCost);

plot(xKDE, yKDE, ...
    'Color', [1.00, 0.45, 0.00], ...
    'LineWidth', 2);

title('Distribution of Painting Costs');
xlabel('Painting Cost (INR)');
ylabel('Density');

legend('Histogram', 'KDE', ...
    'Location', 'northeast');

grid on;
box on;

set(gca, ...
    'FontSize', 11, ...
    'GridLineStyle', '--', ...
    'GridAlpha', 0.3);

hold off;

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/target_distribution.png', ...
    'Resolution', 300);

fprintf('Figure saved to figures/target_distribution.png\n');