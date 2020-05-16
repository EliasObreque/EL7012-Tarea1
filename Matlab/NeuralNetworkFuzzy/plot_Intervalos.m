function []=plot_Intervalos(y,yu,yl, y_ext)

%Parametros de las funciones
        options.handle     = figure();
        options.color_area = [128 193 219]./255;    % Blue theme
        options.color_line = [ 52 148 186]./255;
        %options.color_area = [243 169 114]./255;    % Orange theme
        %options.color_line = [236 112  22]./255;
        options.alpha      = 0.5;
        options.line_width = 0.5;

    if(isfield(options,'x_axis')==0), options.x_axis = 1:size(y,2); end
    options.x_axis = options.x_axis(:);
    
    % Plotting the result
    figure(options.handle);
    x_vector = [options.x_axis', fliplr(options.x_axis')];
    patch = fill(x_vector, [yu,fliplr(yl)], options.color_area);
    set(patch, 'edgecolor', 'none');
    set(patch, 'FaceAlpha', options.alpha);
    hold on;
    grid on;
    plot(options.x_axis,y,'color', options.color_line, ...
        'LineWidth', options.line_width);
    plot(y_ext, '.')
    xlim([0, 500])
    xlabel('Número de muestras')
    ylabel('Salida del modelo')
%     axes('position',[0.62,0.63,0.3058,0.32])
%     box on % put box around new pair of axes
%     x_vector = [options.x_axis', fliplr(options.x_axis')];
%     patch = fill(x_vector, [yu,fliplr(yl)], options.color_area);
%     set(patch, 'edgecolor', 'none');
%     set(patch, 'FaceAlpha', options.alpha);
%     plot(options.x_axis,y,'color', options.color_line, ...
%         'LineWidth', options.line_width)
%     hold on
%     plot(y_ext, '.')
%     xlim([0, 100])
%     grid on
    hold off;
    
end