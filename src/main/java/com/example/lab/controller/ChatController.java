package com.example.lab.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

/**
 * 对话接口：演示"会话记忆 + RAG 检索"两个顾问协作的最小用法。
 *
 * <p>一次请求的完整链路：
 * <pre>
 * 用户提问 → [记忆顾问]拼上历史对话 → [RAG顾问]检索 Milvus 注入知识片段 → 大模型生成回答
 * </pre>
 */
@RestController
@RequestMapping("/chat")
public class ChatController {

    private final ChatClient chatClient;
    private final Advisor ragAdvisor;

    /**
     * chatClient：AiConfig 中已通过 defaultAdvisors 挂上"会话记忆"顾问（所有请求全局生效）；
     * ragAdvisor：RAG 检索顾问（见 RagConfig 的 Bean 定义）。
     * RAG 在这里按请求挂载而不是放进 defaultAdvisors，通过方法进行选择是否需要 RAG
     * 是为了让"哪些接口走 RAG"由控制器自己决定（/multimodal 图片对话就不需要检索）。
     */
    public ChatController(ChatClient chatClient,
                          Advisor ragAdvisor) {
        this.chatClient = chatClient;
        this.ragAdvisor = ragAdvisor;
    }

    @GetMapping("/ask")
    public String ask(@RequestParam String qus,
                      @RequestParam String sessionId) {
        String ans = chatClient.prompt()
                .user(qus)
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .advisors(ragAdvisor) // 开启 rag
                .call()
                .content();
        return ans;
    }

    @GetMapping("/stream")
    public Flux<String> stream(@RequestParam String qus,
                               @RequestParam String sessionId) {
        return chatClient.prompt()
                .user(qus)
                .advisors(ad -> ad.param(ChatMemory.CONVERSATION_ID, sessionId))
                .advisors(ragAdvisor)
                .stream()
                .content();
    }
}
