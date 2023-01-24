from nose.tools import *
from spotad.timeutils import hour_of_week


def test_hour_of_week():

    input1 = ("2017-05-16T17:23:33.572Z","-5")
    input2 = ("2017-05-17T17:47:12.574Z","-5")
    input3 = ("2013-05-12T03:47:12.574Z", "0")
    input4 = ("2013-05-12T03:47:12.574Z", None)

    assert_equals(hour_of_week(*input1), "60")
    assert_equals(hour_of_week(*input2), "84")
    assert_equals(hour_of_week(*input3), hour_of_week(*input4))
