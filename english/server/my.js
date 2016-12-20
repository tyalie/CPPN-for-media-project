$("document").ready(function() {
    reload();
})

function reload() {
    $.ajax({
    	url: "./c.txt?t=" +new Date().getTime() ,
    	async: false,
    	success: function(data) {
            $('body').css("background-color", String(data).replace("0x", "#").replace("L",""));
    	}
    });
    $("img").attr("src","out.png?t="+ new Date().getTime() );
    setTimeout(reload, 500);
}


