using System.ComponentModel;

namespace TestGetDescription
{
    /**
     * 订单状态
     */
    public enum OrderStateEnum
    {
        [Description("待支付")]
        WaitPay,
        [Description("待发货")]
        WaitSend,
        [Description("待收货")]
        WaitReceive,
        [Description("待评论")]
        WaitComment,
        [Description("已完成")]
        Finish,
        [Description("取消订单")]
        Cancel
    }
}
