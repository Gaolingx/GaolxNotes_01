using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DamageZone : MonoBehaviour
{
    // ÿ����Ѫ��
    public int damageNum=-1;

    //�����ڴ������ڵ�ÿһ֡������ô˺������������ڸ���ս���ʱ������һ�Ρ�
    private void OnTriggerStay2D(Collider2D other)
    {
        RubyController rubyController = other.GetComponent<RubyController>();

        if (rubyController != null) {
            rubyController.ChangeHealth(damageNum);

        }
        
    }
}
