def figure_instruction_generator(*args, figure_kwargs = {}):
    packed_data = {}

    # Packaging figure instructions.
    if len(figure_kwargs) == 0:
        figure_kwargs = {'layout': (1, 1)}
    elif figure_kwargs.get('layout') is None:
        figure_kwargs['layout'] = (1, 1)

    packed_data['figure'] = figure_kwargs

    # Packaging plot instructions
    packed_data['plots'] = []
    found_preamble = False
    for subplot in args:
        plot_instruction = {'preamble': {}, 'commands': []}
        for subplot_instruction in subplot:
            '''
            Preamble dict found.
            '''
            if subplot_instruction.get('plot_type') is None and not found_preamble:
                found_preamble = True
                plot_instruction['preamble'] = subplot_instruction
                continue

            if subplot_instruction.get('plot_type') is None and found_preamble:
                raise Exception('Multiple preambles found!')

            '''
            Only the instruction of the plot.
            '''
            plot_instruction['commands'].append(subplot_instruction)

        packed_data['plots'].append(plot_instruction)

    return packed_data
