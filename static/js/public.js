$(function() {
	$.get("https://v1.hitokoto.cn", null,
	function(a) {
		$(".intro-siteinfo").html(a.hitokoto + " —— <strong>" + a.from + "</strong>")
	},
	"JSON"),
	window.onscroll = function() {
		var b = 0,
		c = 0,
		a = 0;
		document.body && (c = document.body.scrollTop),
		document.documentElement && (a = document.documentElement.scrollTop),
		b = c - a > 0 ? c: a,
		b > 0 ? $("#go-to-top").show() : $("#go-to-top").hide()
	},
	$("#go-to-top").click(function() {
		$("html,body").animate({
			scrollTop: 0
		},
		250)
	})
});
$("footer").before('<a class="am-icon-btn am-icon-arrow-up am-active" id="go-to-top"></a>');