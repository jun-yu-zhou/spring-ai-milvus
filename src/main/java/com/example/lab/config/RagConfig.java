package com.example.lab.config;

import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * RAG（Retrieval-Augmented Generation，检索增强生成）配置。
 *
 * <p>RAG 解决的问题：大模型不了解你的私有知识，直接问会"一本正经地编"（幻觉）。
 * 做法是：回答之前先去 Milvus 检索出与问题语义最相关的知识片段，
 * 把片段拼进提示词，让模型"看着资料回答"，而不是凭空编造。
 *
 * <p>这里用的是 Spring AI 的"模块化 RAG"顾问 RetrievalAugmentationAdvisor（在
 * spring-ai-rag 模块中），它把 RAG 流程拆成 查询改写 → 检索 → 文档合并
 * 三段可插拔的积木；本配置只替换了"检索"这一段（documentRetriever），其余走默认。
 * 它挂到对话请求后自动执行：
 * <ol>
 *   <li>拿用户问题去 Milvus 做 similaritySearch：最多取 5 片，相似度低于 0.5 的丢弃；</li>
 *   <li>把保留下来的片段作为 context 注入提示词，随用户问题一起发给大模型。</li>
 * </ol>
 *
 * <p>相对的还有更简单的 QuestionAnswerAdvisor（"朴素 RAG"，一行代码开箱即用，
 * 在 spring-ai-vector-store-advisor 模块，适合 MVP 快速验证）。
 * 两者二选一即可，本项目选用前者以便后续定制检索流程。
 */
@Configuration
public class RagConfig {

    private final VectorStore milvusVectorStore;

    public RagConfig(VectorStore milvusVectorStore) {
        this.milvusVectorStore = milvusVectorStore;
    }

    /**
     * RAG 检索顾问 Bean：设置为顾问类型，以 {@link Advisor} 类型暴露，由 ChatController 构造器注入，
     * 再通过 chatClient.prompt().advisors(ragAdvisor) 按请求挂载——
     * 这样"哪些接口走 RAG"由控制器自己决定（例如 /multimodal 图片对话就不需要检索）。
     */
    @Bean
    public Advisor ragAdvisor() {
        return RetrievalAugmentationAdvisor.builder()
                // "检索"积木：VectorStoreDocumentRetriever 封装了"问题向量化 → Milvus 相似度搜索"
                .documentRetriever(
                        VectorStoreDocumentRetriever.builder()
                                .vectorStore(milvusVectorStore)
                                .similarityThreshold(0.5) // 相似度门槛：低于 0.5 的片段直接丢弃，宁缺毋滥
                                .topK(5)                  // 最多取 5 片：太多会撑爆上下文，还会稀释重点
                                .build()
                )
                .build();
    }
}
