$(function () {
    $("#knn_btn").click(function (event) {
        $("#floatingCirclesG").show();
        event.preventDefault();
        var knn_btn = $('#knn_btn').attr("id");
        $.post({
            'url': '/algorithm/',
            'data': {
                'method': knn_btn,
            },
            'success': function (data) {
                if (data['code'] == 200) {
                    var message = data['message'];
                    $("#message-knn").html(message);
                    $("#message-knn").show();
                    document.getElementById('knn_pic').src = data.img_path;
                    console.log("succ");
                    $("#floatingCirclesG").hide();
                }
                console.log(data);
            },
            'fail': function (error) {
                console.log(error);
            }
        });
    });


    $("#svm_btn").click(function (event) {
        event.preventDefault();
        var svm_btn = $('#svm_btn').attr("id");
        $.post({
            'url': '/algorithm/',
            'data': {
                'method': svm_btn
            },
            'success': function (data) {
                if (data['code'] == 200) {
                    var message = data['message'];
                    $("#message-svm").html(message);
                    $("#message-svm").show();
                }
                console.log(data);
            },
            'fail': function (error) {
                console.log(error);
            }
        });

    });

    $("#nbayes_btn").click(function (event) {
        event.preventDefault();
        var nbayes_btn = $('#nbayes_btn').attr("id");
        $.post({
            'url': '/algorithm/',
            'data': {
                'method': nbayes_btn
            },
            'success': function (data) {
                if (data['code'] == 200) {
                    var message = data['message'];
                    $("#message-nbayes").html(message);
                    $("#message-nbayes").show();
                }
                console.log(data);
            },
            'fail': function (error) {
                console.log(error);
            }
        });

    });

    $("#address_btn").click(function (event) {
        event.preventDefault();
        var address_btn = $('#address_btn').attr("id");
        var address = $('#address').val();
        $.post({
            'url': '/nlp/',
            'data': {
                'method': address_btn,
                'address': address
            },
            'success': function (data) {
                if (data['code'] == 200) {
                    var message = data['message'];
                }
                console.log(data);
            },
            'fail': function (error) {
                console.log(error);
            }
        });

    });

});