using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HealthCollectible : MonoBehaviour
{
    //��ݮ�ӵ�Ѫ��
    public int amount=1;
    //������¼��ײ����
    int collideCount;
    //��Ӵ�������ײ�¼���ÿ����ײ������ʱ��ִ�����еĴ���
    private void OnTriggerEnter2D(Collider2D other)
    {
        collideCount = collideCount + 1;
        Debug.Log($"�͵�ǰ���巢����ײ�Ķ����ǣ�{other}����ǰ�ǵ�{collideCount}����ײ��");

        //��ȡ Ruby ��Ϸ����Ľű��������
        RubyController rubyController = other.GetComponent<RubyController>();

        if (rubyController != null)
        {
            if (rubyController.health < rubyController.maxHealth)
            {
                //��������ֵ
                rubyController.ChangeHealth(amount);
                //���ٵ�ǰ��Ϸ����
                //�����ò�ݮ���Ե�����ʧ
                Destroy(gameObject);
            }
            else {
                Debug.Log("��ǰ��ҽ�ɫ���������ģ�����Ҫ��Ѫ��");
            }
        }
        else {
            Debug.LogError("rubyController ��Ϸ�����δ��ȡ������������");
        }

    }
}
