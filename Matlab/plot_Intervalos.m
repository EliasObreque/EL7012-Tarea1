function []=plot_Intervalos(y,yu,yl)

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
    plot(options.x_axis,y,'color', options.color_line, ...
        'LineWidth', options.line_width);
    hold off;
    
     xlabel('t')
     ylabel('Salida del modelo')
     xlim([168 288])
end