(function(IPython){
    var format_block = function(cell, evt){      
        var block_types = { TIP: "fa-lightbulb-o", WARNING: "fa-warning", 
        LINK: "fa-link", SOURCE: "fa-file-text", QUESTION: "fa-question"};
        var cell = evt.cell;
        if(cell.get_rendered == undefined){
            return;
        }
        var block = jQuery(cell.get_rendered());
        if(block.prop("tagName") == "BLOCKQUOTE"){
            var mark = block.find("p strong").text();
            if(block_types[mark] != undefined){
                var node = cell.element.find('div.text_cell_render');
                var content = node.find("p:not(:first)");
                cell.element.find("blockquote").addClass("info_block")
                .html('<table class="info_block"><tr><td class="first-column"></td><td></td></tr></table>');
                cell.element.find("td:first").html('<div class="fa large_font ' + block_types[mark] + '"></div>');
                cell.element.find("td:last").append(content);
            }
        }
    };
    
    jQuery([IPython.events]).on('create.Cell', function(notebook, evt){
        var cell = evt.cell;
        var index = evt.index;
        cell.events.on('rendered.MarkdownCell', format_block);
    });
    
    IPython.notebook.get_cells().map(function (cell, i) {
        cell.events.on('rendered.MarkdownCell', format_block);
        cell.events.trigger("rendered.MarkdownCell", {cell: cell});
    });

    //$("head").append($("<link rel='stylesheet' href='/static/custom/usability/execute_time/ExecuteTime.css' type='text/css' />"));

}(IPython));