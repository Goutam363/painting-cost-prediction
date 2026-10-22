models = {'LR', 'DT'};

training_energy = [0.253208, 0.155631];
testing_energy = [0.101509, 0.098981];

total_energy = training_energy + testing_energy;

assignin('base', 'models', models);
assignin('base', 'training_energy', training_energy);
assignin('base', 'testing_energy', testing_energy);
assignin('base', 'total_energy', total_energy);

fprintf('\nENERGY CONSUMPTION\n');
for i = 1:numel(models)
    fprintf('%s: %.4f J\n', models{i}, total_energy(i));
end

figure('Color', 'w', 'Position', [100, 100, 900, 500]);

bars = bar(total_energy, ...
    0.6, ...
    'FaceColor', [0.12, 0.47, 0.71], ...
    'EdgeColor', 'black');

xtips = bars.XEndPoints;
ytips = bars.YEndPoints;

labels = compose('%.4f J', total_energy);

text(xtips, ytips, labels, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'FontSize', 11);

title('Energy Consumption Comparison');
xlabel('Machine Learning Model');
ylabel('Energy Consumption (J, log scale)');

xticks(1:numel(models));
xticklabels(models);

set(gca, 'YScale', 'log');

grid on;

ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.3;
ax.Layer = 'bottom';

ylim([0.15, 0.75]);

set(gca, 'FontSize', 11);

if ~exist('figures', 'dir')
    mkdir('figures');
end

exportgraphics(gcf, ...
    'figures/energy_comparison.png', ...
    'Resolution', 300);

fprintf('\nFigure saved to figures/energy_comparison.png\n');