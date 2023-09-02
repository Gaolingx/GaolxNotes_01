using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyController : MonoBehaviour
{
    //�趨�ƶ��ٶȱ���
    public float speed = 0.1f; 
    //����һ��2d�������
    Rigidbody2D rigidbody2d;
    // ���� Vector2 ��������ŵ��˵�ǰλ��
    Vector2 position;
    //����һ����ʼ y �������
    float initY;
    //����һ���ƶ�����ı���
    float direction;
    //����ƶ�����ı���������Ϊ���У������� unity �еķ���
    public float distance=4;

    // Start is called before the first frame update
    void Start()
    {
        // ��ȡ��Щ������������Ϸ��ʼʱ��ֵ
        rigidbody2d = GetComponent<Rigidbody2D>();
        //��ȡ��ʼλ��
        position = transform.position;
        //��ȡ��ʼy����
        initY = position.y;
        //�趨��ʼ�ƶ�����
        direction = 1.0f;
    }

    private void FixedUpdate()
    {
        //ͨ�������ƶ��ķ������ã����� fixupdate�����У�0.02��ִ��һ��
        MovePosition();
    }

    // �Զ������ Y ���۷��ƶ����㷨
    private void MovePosition() {
        if (position.y-initY< distance && direction>0)
        {
            position.y += speed;
        }
        if (position.y - initY >= distance && direction > 0)
        {
            direction = -1.0f;
        }
        if (position.y - initY > 0&&direction<0)
        {
            position.y -= speed;
        }
        if (position.y - initY <= 0 && direction < 0)
        {
            direction = 1.0f;
        }
        //ͨ������ϵͳ�ƶ���Ϸ����
        rigidbody2d.position = position;
    }
}
