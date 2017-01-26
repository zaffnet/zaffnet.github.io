ZeroClipboard.config({
    hoverClass: 'btn-clipboard-hover'
})
$('.highlight').each(function () {
    var btnHtml = '<div class="zero-clipboard"><span class="btn-clipboard"><img  width="13" src="assets/clippy.svg" alt="Copy to clipboard"></span></div>';
    $(this).before(btnHtml)
});
var zeroClipboard = new ZeroClipboard($('.btn-clipboard'));
var htmlBridge = $('#global-zeroclipboard-html-bridge');
zeroClipboard.on('ready', function (event) {
    htmlBridge
        .data('placement', 'top')
        .attr('title', 'Copy to clipboard')
        .tooltip();


    zeroClipboard.on('copy', function (event) {
        var highlight = $(event.target).parent().nextAll('.highlight').first().find('.code').first();
        if (highlight.length == 0)
        {
          highlight = $(event.target).parent().nextAll('.highlight').first().find('code').first();
        }
        event.clipboardData.setData("text/plain", highlight.text())
    });
    zeroClipboard.on('aftercopy', function () {
        htmlBridge
            .attr('title', 'Copied!')
            .tooltip('fixTitle')
            .tooltip('show')
            .attr('title', 'Copy to clipboard')
            .tooltip('fixTitle')
    });
});

zeroClipboard.on('error', function () {
    ZeroClipboard.destroy();
    htmlBridge
        .attr('title', 'Flash required')
        .tooltip('fixTitle')
        .tooltip('show');
});
$(document).ready(function () {
	$( ".highlighttable" ).wrap("<div class='table-responsive'></div>");
});

(function(){
	var d = document,
	accordionToggles = d.querySelectorAll('.js-accordionTrigger'),
	setAria,
	setAccordionAria,
	switchAccordion,
  touchSupported = ('ontouchstart' in window),
  pointerSupported = ('pointerdown' in window);

  skipClickDelay = function(e){
    e.preventDefault();
    e.target.click();
  }
		setAriaAttr = function(el, ariaType, newProperty){
		el.setAttribute(ariaType, newProperty);
	};
	setAccordionAria = function(el1, el2, expanded){
		switch(expanded) {
      case "true":
      	setAriaAttr(el1, 'aria-expanded', 'true');
      	setAriaAttr(el2, 'aria-hidden', 'false');
      	break;
      case "false":
      	setAriaAttr(el1, 'aria-expanded', 'false');
      	setAriaAttr(el2, 'aria-hidden', 'true');
      	break;
      default:
				break;
		}
	};
switchAccordion = function(e) {
  console.log("triggered");
	e.preventDefault();
	var thisAnswer = e.target.parentNode.nextElementSibling;
	var thisQuestion = e.target;
	if(thisAnswer.classList.contains('is-collapsed')) {
		setAccordionAria(thisQuestion, thisAnswer, 'true');
	} else {
		setAccordionAria(thisQuestion, thisAnswer, 'false');
	}
  	thisQuestion.classList.toggle('is-collapsed');
  	thisQuestion.classList.toggle('is-expanded');
		thisAnswer.classList.toggle('is-collapsed');
		thisAnswer.classList.toggle('is-expanded');

  	thisAnswer.classList.toggle('animateIn');
	};
	for (var i=0,len=accordionToggles.length; i<len; i++) {
		if(touchSupported) {
      accordionToggles[i].addEventListener('touchstart', skipClickDelay, false);
    }
    if(pointerSupported){
      accordionToggles[i].addEventListener('pointerdown', skipClickDelay, false);
    }
    accordionToggles[i].addEventListener('click', switchAccordion, false);
  }
})();
jQuery(document).ready(function() {
var offset = 250;
var duration = 300;
jQuery(window).scroll(function() {
if (jQuery(this).scrollTop() > offset) {
jQuery('.back-to-top').fadeIn(duration);
} else {
jQuery('.back-to-top').fadeOut(duration);
}});
jQuery('.back-to-top').click(function(event) {
event.preventDefault();
jQuery('html, body').animate({scrollTop: 0}, duration);
return false;})});
