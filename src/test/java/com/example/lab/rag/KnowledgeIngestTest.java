package com.example.lab.rag;

import com.example.lab.service.KnowledgeService;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

import java.io.IOException;
import java.util.List;

/**
 * 知识库"入库 → 检索"端到端验证（连接真实的 Milvus 和百炼 Embedding 模型）。
 *
 * <p>运行前提：Milvus(19530) 可用、application-dev.yml 中的 api-key 有效。
 * 首次运行时 initialize-schema=true 会自动创建 vector_store 集合和索引。
 */
@SpringBootTest
class KnowledgeIngestTest {

    @Autowired
    KnowledgeService knowledgeService;
    @Autowired
    VectorStore vectorStore;

    /**
     * 用例一：纯文本入库 → 相似度检索，最小可跑通的一套流程。
     * 真实项目里把第 2 步换成 PDF 读取器的产出即可（见用例二），其余步骤完全一样。
     */
    @Test
    void textIngestAndSearch() {
        // 1. 入库前先按 file_name 删旧数据，保证测试可重复执行。
        //    向量库不会自动去重：不删的话，重复跑一次就存两份，检索结果翻倍。
        //    这个"过滤表达式"会被 Spring AI 翻译成 Milvus 的 JSON 条件
        //    （metadata["file_name"] == "milvus-说明.txt"）。
        vectorStore.delete("file_name == 'milvus-说明.txt'");

        // 2. 纯文本入库：内部经历"切分 → 向量化 → 写入 Milvus"
        knowledgeService.ingest("milvus-说明.txt", "txt", """
                Milvus 是一款开源的高性能向量数据库，专为海量向量的相似度检索而设计。
                它支持 HNSW、IVF_FLAT 等多种索引类型，通常用 COSINE 余弦相似度来衡量向量之间的距离。
                在 RAG 架构中，Milvus 负责存储文档切片生成的向量，当用户提问时，
                它能快速找出语义最相近的若干片段，交给大模型生成有依据的回答。
                """);

        // 3. 端到端验证：问题会被自动向量化 → Milvus 相似度检索 → 应命中刚才入库的片段
        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query("RAG 里向量数据库负责什么")
                        .topK(3)
                        .build());
        System.out.println("命中条数: " + hits.size());
        hits.forEach(d -> System.out.println(
                d.getMetadata().get("file_name") + " | " + d.getText()));
    }

    /**
     * 用例二：PDF 入库。PDF 统一放在 src/main/resources/knowledge/ 目录（本用例
     * 通过 classpath 扫描该目录——main 资源在测试运行时同样位于 classpath 上）；
     * 目录为空时自动跳过（Assumptions.assumeTrue），不影响其他用例。
     *
     * <p>PagePdfDocumentReader 每页产出一个 Document，自带
     * file_name / page_number 元数据——检索命中后就能告诉用户"答案来自第几页"。
     */
    @Test
    void pdfIngestAndSearch() throws IOException {
        // 扫描 classpath 上的 knowledge 目录（即 src/main/resources/knowledge/）
        Resource[] pdfs = new PathMatchingResourcePatternResolver()
                .getResources("classpath:knowledge/*.pdf");
        Assumptions.assumeTrue(pdfs.length > 0, "knowledge 目录下没有 PDF，跳过本用例");

        for (Resource pdf : pdfs) {
            String fileName = pdf.getFilename();
            // 同样先删旧数据保证可重复执行
            vectorStore.delete("file_name == '" + fileName + "'");
            // 逐页读取（E 提取），交给 KnowledgeService 切分入库（T + L）
            List<Document> pages = new PagePdfDocumentReader(pdf).get();
            knowledgeService.ingest(pages);
        }

        List<Document> hits = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query("文档的主要内容是什么")
                        .topK(3)
                        .build());
        hits.forEach(d -> System.out.println(
                d.getMetadata().get("file_name") + " | " + d.getText()));
    }
}
