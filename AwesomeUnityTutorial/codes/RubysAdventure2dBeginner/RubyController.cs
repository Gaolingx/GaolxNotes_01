using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RubyController : MonoBehaviour
{
    //��������޵е�ʱ����
    public float timeInvincible = 2.0f;
    // �����Ƿ��޵еı���
    bool isInvincible;
    // ��������������޵�ʱ��ļ�ʱ���޵�ʱ���ʱ��
    float invincibleTimer;

    // �����������ֵ���������ޣ�
    public int maxHealth = 5;
    // ���õ�ǰ����ֵ������ health

    // C# ��֧����������������еķ�װ��������ݳ�Ա�ı���
    // ���ݳ�Ա������Ĭ��һ�㶼Ӧ������Ϊ˽�У�ֻ��ͨ����ǰ��ķ��������Խ��з���
    // �����ǹ��еģ�����ͨ��ȡֵ�� get ����ֵ�� set �趨��Ӧ�ֶεķ��ʹ������������ܹ�����
    public int health { 
        get { return currentHealth; } 
        //set {currentHealth=value;}
    }
    int currentHealth; 

    //�����������
    Rigidbody2D rigidbody2d;
    //��ȡ�û�����
    float horizontal;
    float vertical;

    // ���ٶȱ�¶������ʹ��ɵ�
    public float speed = 0.1f;

    //����һ�� �����������������
    Animator animator;
    //����һ����άʸ���������洢 Ruby ��ֹ���ƶ�ʱ ���ķ��򣨼�����ķ���
    //���������ȣ�Ruby ����վ����������վ������ʱ��Move X �� Y ��Ϊ 0����ʱ״̬����û����ȡ Ruby ��ֹʱ�ĳ���
    //������Ҫ�ֶ�����һ��
    Vector2 lookDirection = new Vector2(1, 0);
    Vector2 move;
    private void Start()
    {
        //��ȡ��ǰ��Ϸ����ĸ������
        rigidbody2d = GetComponent<Rigidbody2D>();
        //��Ϸ�տ�ʼ�������Ѫ
        //��ʼ����ǰ����ֵ
        currentHealth = maxHealth;

        animator = GetComponent<Animator>();
    }

    // ÿ֡����һ�� Update
    void Update()
    {
        horizontal = Input.GetAxis("Horizontal");
        vertical = Input.GetAxis("Vertical");

        //�ж��Ƿ����޵�״̬�������м�ʱ���ĵ���ʱ
        if (isInvincible) {
            //����޵У����뵹��ʱ
            //invincibleTimer = invincibleTimer - Time.deltaTime;
            //ÿ��update��ȥһ֡�����ĵ�ʱ��
            invincibleTimer -= Time.deltaTime;
            //ֱ����ʱ����ʱ������
            if (invincibleTimer < 0) {
                //ȡ���޵�״̬
                isInvincible = false;
            }
        }
        
        //����һ����άʸ����������ʾ Ruby�ƶ���������Ϣ
        move = new Vector2(horizontal, vertical);
        //���move�е� x/y ��Ϊ�㣬��ʾ�����˶�
        //�� ruby ����������Ϊ�ƶ�����
        //ֹͣ�ƶ���������ǰ����������� If �ṹ����ת��ʱ���¸�ֵ�泯����
        if (!Mathf.Approximately(move.x, 0.0f) || !Mathf.Approximately(move.y, 0.0f)) {
            lookDirection.Set(move.x, move.y);

            //ʹ��������Ϊ1�����Խ��˷�����Ϊ �����ġ���һ��������
            //ͨ�����ڱ�ʾ���򣬶���λ�õ�������
            //��Ϊblend tree �б�ʾ����Ĳ���ֵȡֵ��Χ�� -1.0 �� 1.0��
            //����һ���� ������Ϊ Animator.SetFloat �����Ĳ���ʱ��һ��Ҫ�������Ƚ��� ����һ�������� 
            lookDirection.Normalize();
        }
        
        //���� Ruby �泯���� �� blend tree
        animator.SetFloat("Look X", lookDirection.x);
        animator.SetFloat("Look Y", lookDirection.y);
        //���� Ruby �ٶȸ� blend tree
        //ʸ���� magnitue ���ԣ���������ʸ���ĳ���
        animator.SetFloat("Speed", move.magnitude);

    }

    //�̶�ʱ����ִ�еĸ��·���
    private void FixedUpdate()
    {
        Vector2 position = transform.position;
        position= position + speed * move * Time.deltaTime;
        
        rigidbody2d.MovePosition(position);
    }

    //��������ֵ�ķ���
    public void ChangeHealth(int amount) {
        //����������˺���ʱ������������2��
        if (amount < 0) {
            //�жϵ�ǰ����Ƿ����޵�״̬
            if (isInvincible) {
                //�޵�״̬����Ѫ��������ǰ����
                return;
            }
            //�������޵�״̬ʱ����ִ������Ĵ���
            //�����޵�״̬Ϊ��
            isInvincible = true;
            //�����޵�ʱ��
            invincibleTimer = timeInvincible;

            //�������˶���
            animator.SetTrigger("Hit");

        }


        //���Ʒ��������Ƶ�ǰ����ֵ�ĸ�ֵ��Χ��0-�������ֵ
        currentHealth = Mathf.Clamp(currentHealth + amount, 0, maxHealth);
        //�ڿ���̨���������Ϣ
        Debug.Log("��ǰ����ֵ�� " + currentHealth + "/" + maxHealth);
    }

}
